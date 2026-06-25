import numpy as np


class WGAN:
    def __init__(
        self,
        k=5,
        random_state=None,
        epochs=300,
        batch_size=64,
        noise_dim=32,
        lr=1e-4,
        gp_lambda=10.0,
        critic_iters=5,
        *,
        device=None,
    ):
        self.k = k
        self.random_state = (
            np.random.RandomState(random_state)
            if random_state is not None
            else np.random.RandomState()
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.lr = lr
        self.gp_lambda = gp_lambda
        self.critic_iters = critic_iters
        self._device = device

    def fit_resample(self, X, y):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        device = self._device or torch.device("cpu")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        minority_count = counts.min()
        majority_count = counts.max()

        n_to_generate = majority_count - minority_count
        if n_to_generate <= 0:
            return X, y

        X_min = X[y == minority_class]
        n_features = X.shape[1]

        torch.manual_seed(self.random_state.randint(0, 2 ** 31 - 1))
        X_tensor = torch.tensor(X_min, dtype=torch.float32, device=device)
        n_real = X_tensor.size(0)
        batch_size = min(self.batch_size, n_real)

        class _Generator(nn.Module):
            def __init__(self, noise_dim, output_dim, hidden=256):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(noise_dim, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Linear(hidden // 2, output_dim),
                )

            def forward(self, z):
                return self.net(z)

        class _Discriminator(nn.Module):
            def __init__(self, input_dim, hidden=256):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden, hidden),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden, hidden // 2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden // 2, 1),
                )

            def forward(self, x):
                return self.net(x)

        G = _Generator(self.noise_dim, n_features).to(device)
        D = _Discriminator(n_features).to(device)
        opt_G = optim.Adam(G.parameters(), lr=self.lr, betas=(0.0, 0.9))
        opt_D = optim.Adam(D.parameters(), lr=self.lr, betas=(0.0, 0.9))

        G.train()
        D.train()

        for _ in range(self.epochs):
            for _ in range(self.critic_iters):
                idx = torch.randint(0, n_real, (batch_size,))
                real_batch = X_tensor[idx]
                z = torch.randn(batch_size, self.noise_dim, device=device)

                D.zero_grad()
                fake = G(z).detach()
                d_real = D(real_batch)
                d_fake = D(fake)

                alpha = torch.rand(batch_size, 1, device=device).expand_as(real_batch)
                interpolated = alpha * real_batch + (1 - alpha) * fake
                interpolated.requires_grad_(True)
                d_interpolated = D(interpolated)
                grads = torch.autograd.grad(
                    outputs=d_interpolated, inputs=interpolated,
                    grad_outputs=torch.ones_like(d_interpolated),
                    create_graph=True, retain_graph=True,
                )[0]
                gp = ((grads.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda

                d_loss = d_fake.mean() - d_real.mean() + gp
                d_loss.backward()
                opt_D.step()

            G.zero_grad()
            z = torch.randn(batch_size, self.noise_dim, device=device)
            fake = G(z)
            g_loss = -D(fake).mean()
            g_loss.backward()
            opt_G.step()

        G.eval()
        with torch.no_grad():
            n_batches = int(np.ceil(n_to_generate / 1024))
            synth_parts = []
            for _ in range(n_batches):
                n_left = n_to_generate - sum(p.size(0) for p in synth_parts)
                z = torch.randn(min(1024, n_left), self.noise_dim, device=device)
                synth_parts.append(G(z).cpu().numpy())
            X_synth = np.vstack(synth_parts).astype(np.float64)
            y_synth = np.full(len(X_synth), minority_class)

        X_resampled = np.vstack([X, X_synth[:n_to_generate]])
        y_resampled = np.hstack([y, y_synth[:n_to_generate]])
        return X_resampled, y_resampled
