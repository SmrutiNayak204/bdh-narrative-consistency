import torch
from bdh import BDH

class BDHWithState(BDH):
    def forward(self, idx, targets=None, return_state=False):
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)

        for _ in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = torch.relu(x_latent)

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = torch.relu(y_latent)

            xy_sparse = x_sparse * y_sparse
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, -1) @ self.decoder
            )

            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head

        if return_state:
            return logits, x.detach()

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss
   