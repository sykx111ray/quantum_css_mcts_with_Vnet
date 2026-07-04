"""
matrix_encoder.py — Row/Column Transformer Matrix Encoder

Encodes a GF(2) binary stabilizer matrix into a fixed-dimension embedding,
preserving 2-D row-column structure rather than flattening.

Architecture (following axial attention / row-column transformer):
  Matrix (R×C binary)
     ↓
  Row Embedding: per-bit embed → mean over columns → + row position
     ↓
  Row Transformer: nn.TransformerEncoder over R row tokens
     ↓
  Column Embedding: mat^T @ row_out → aggregated per column → / deg → + col position
     ↓
  Column Transformer: nn.TransformerEncoder over C column tokens
     ↓
  Pooling: concat(mean(row_out), mean(col_out)) → Linear → embedding

All components are standard PyTorch (nn.TransformerEncoder, nn.Embedding, nn.Linear).
Supports variable-size matrices via dynamic embedding (no fixed Linear input dim).
"""
import torch
import torch.nn as nn


class MatrixEncoder(nn.Module):
    """Row/Column Transformer for binary stabilizer matrices.

    Args:
        embed_dim:   Token embedding dimension (output dim = embed_dim).
        nhead:       Number of attention heads in TransformerEncoder.
        num_layers:  Number of TransformerEncoder layers for each of row/col.
        max_size:    Max rows or columns (for learnable position embeddings).
        dropout:     Dropout rate in Transformer layers.
    """

    def __init__(self, embed_dim=128, nhead=4, num_layers=2,
                 max_size=200, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Per-bit embedding: 0 → vec, 1 → vec
        self.bit_embed = nn.Embedding(2, embed_dim)

        # Learnable position embeddings for rows and columns
        self.row_pos = nn.Embedding(max_size, embed_dim)
        self.col_pos = nn.Embedding(max_size, embed_dim)

        # Row TransformerEncoder
        self.row_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=4 * embed_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Column TransformerEncoder
        self.col_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=4 * embed_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Output: concat(row_pool, col_pool) → embed_dim
        self.out_norm = nn.LayerNorm(2 * embed_dim)
        self.out_proj = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, matrix):
        """Encode a batch of binary matrices.

        Args:
            matrix: (B, R, C) tensor with values in {0, 1} (int, bool, or float).
                    R and C may vary across batches but must be <= max_size.

        Returns:
            (B, embed_dim) tensor — matrix embedding.
        """
        B, R, C = matrix.shape
        D = self.embed_dim
        device = matrix.device

        # Ensure integer {0, 1}
        mat_long = matrix.long().clamp(0, 1)

        # ================================================================
        # 1. Row Embedding
        # ================================================================
        # Per-bit embed: (B, R, C) → (B, R, C, D)
        bits = self.bit_embed(mat_long)
        # Add column position: (1, 1, C, D)
        col_idx = torch.arange(C, device=device)
        bits = bits + self.col_pos(col_idx).unsqueeze(0).unsqueeze(0)
        # Mean over columns → row token: (B, R, D)
        row_tokens = bits.mean(dim=2)
        # Add row position: (1, R, D)
        row_idx = torch.arange(R, device=device)
        row_tokens = row_tokens + self.row_pos(row_idx).unsqueeze(0)

        # ================================================================
        # 2. Row Transformer
        # ================================================================
        # (B, R, D) → (B, R, D)
        row_out = self.row_encoder(row_tokens)

        # ================================================================
        # 3. Column Embedding (informed by row attention)
        # ================================================================
        # Aggregate row_out to columns via matrix transpose multiply.
        # mat: (B, R, C), mat.T: (B, C, R), row_out: (B, R, D)
        # → col_feat: (B, C, D)  [sum of row features for incident rows]
        mat_f = mat_long.float()
        col_tokens = torch.bmm(mat_f.transpose(1, 2), row_out)   # (B, C, D)

        # Normalize by column degree (avoid bias towards high-degree columns)
        col_deg = mat_f.sum(dim=1).clamp(min=1).unsqueeze(-1)     # (B, C, 1)
        col_tokens = col_tokens / col_deg

        # Add column position: (1, C, D)
        col_tokens = col_tokens + self.col_pos(col_idx).unsqueeze(0)

        # ================================================================
        # 4. Column Transformer
        # ================================================================
        # (B, C, D) → (B, C, D)
        col_out = self.col_encoder(col_tokens)

        # ================================================================
        # 5. Pooling & Output
        # ================================================================
        row_pool = row_out.mean(dim=1)              # (B, D)
        col_pool = col_out.mean(dim=1)              # (B, D)
        combined = torch.cat([row_pool, col_pool], dim=-1)  # (B, 2D)

        return self.out_proj(self.out_norm(combined))   # (B, D)
