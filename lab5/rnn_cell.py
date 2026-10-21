import numpy as np


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size, seed=None):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # U(-1/sqrt(H), 1/sqrt(H))
        rng = np.random.default_rng(seed)
        k = 1.0 / np.sqrt(hidden_size)
        self.W_xh = rng.uniform(-k, k, size=(hidden_size, input_size))
        self.W_hh = rng.uniform(-k, k, size=(hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size)

        self.cache = None

    def set_weights(self, W_xh, W_hh, b_h):
        W_xh = np.asarray(W_xh, dtype=float)
        W_hh = np.asarray(W_hh, dtype=float)
        b_h = np.asarray(b_h, dtype=float)
        assert W_xh.shape == (self.hidden_size, self.input_size), "W_xh sai shape"
        assert W_hh.shape == (self.hidden_size, self.hidden_size), "W_hh sai shape"
        assert b_h.shape == (self.hidden_size,), "b_h sai shape"
        self.W_xh, self.W_hh, self.b_h = W_xh, W_hh, b_h

    def init_hidden(self, batch_size=None):
        """Hidden state ban đầu h_0 = 0."""
        if batch_size is None:
            return np.zeros(self.hidden_size)
        return np.zeros((batch_size, self.hidden_size))

    def forward(self, x_t, h_prev=None):
        # (x_t, h_{t-1}) -> h_t
        x_t = np.asarray(x_t, dtype=float)
        if h_prev is None:
            h_prev = self.init_hidden(None if x_t.ndim == 1 else x_t.shape[0])
        h_prev = np.asarray(h_prev, dtype=float)

        if x_t.shape[-1] != self.input_size:
            raise ValueError(f"x_t có chiều cuối {x_t.shape[-1]}, cần {self.input_size}")
        if h_prev.shape[-1] != self.hidden_size:
            raise ValueError(f"h_prev có chiều cuối {h_prev.shape[-1]}, cần {self.hidden_size}")

        # a_t = W_xh x_t + W_hh h_{t-1} + b_h 
        a_t = x_t @ self.W_xh.T + h_prev @ self.W_hh.T + self.b_h
        h_t = np.tanh(a_t)

        self.cache = {"x_t": x_t, "h_prev": h_prev, "a_t": a_t, "h_t": h_t}
        return h_t

    __call__ = forward

    def forward_sequence(self, xs, h0=None):
        xs = np.asarray(xs, dtype=float)
        T = xs.shape[-2]
        h = h0 if h0 is not None else self.init_hidden(None if xs.ndim == 2 else xs.shape[0])
        hs = []
        for t in range(T):
            h = self.forward(xs[..., t, :], h)
            hs.append(h)
        return np.stack(hs, axis=-2), h


def test_bai1():
    x_t = np.array([1.0, 0.0])
    h_prev = np.array([0.5, 0.2])
    W_xh = [[0.5, 0.2], [0.1, 0.4]]
    W_hh = [[0.3, 0.1], [0.2, 0.5]]
    b_h = [0.0, 0.0]

    cell = SimpleRNNCell(input_size=2, hidden_size=2)
    cell.set_weights(W_xh, W_hh, b_h)
    h_t = cell.forward(x_t, h_prev)

    # a_1 = 0.5*1 + 0.2*0 + 0.3*0.5 + 0.1*0.2 = 0.67
    # a_2 = 0.1*1 + 0.4*0 + 0.2*0.5 + 0.5*0.2 = 0.30
    a_expected = np.array([0.67, 0.30])
    assert np.allclose(cell.cache["a_t"], a_expected), cell.cache["a_t"]
    assert np.allclose(h_t, np.tanh(a_expected)), h_t

    print("[Bai 1] a_t =", cell.cache["a_t"])
    print("[Bai 1] h_t =", np.round(h_t, 4))


def compare_with_pytorch(input_size=4, hidden_size=3, batch=5, T=7, seed=0):
    """So sánh với torch.nn.RNNCell và torch.nn.RNN (tanh)."""
    try:
        import torch
    except ImportError:
        print("Check pytorch")
        return

    rng = np.random.default_rng(seed)
    cell = SimpleRNNCell(input_size, hidden_size, seed=seed)
    cell.b_h = rng.normal(size=hidden_size)  # bias khác 0 

    # Một bước: torch.nn.RNNCell
    # x = rng.normal(size=(batch, input_size))
    # h = rng.normal(size=(batch, hidden_size))
    x = rng.normal(size=(batch, input_size)).astype(np.float32)
    h = rng.normal(size=(batch, hidden_size)).astype(np.float32)

    t_cell = torch.nn.RNNCell(input_size, hidden_size, nonlinearity="tanh")
    with torch.no_grad():
        t_cell.weight_ih.copy_(torch.from_numpy(cell.W_xh))
        t_cell.weight_hh.copy_(torch.from_numpy(cell.W_hh))
        t_cell.bias_ih.copy_(torch.from_numpy(cell.b_h))
        t_cell.bias_hh.zero_()  # PyTorch có 2 bias (b_ih + b_hh); chỉ dùng 1 b_h
        h_torch = t_cell(torch.from_numpy(x), torch.from_numpy(h)).numpy()

    h_mine = cell.forward(x, h)
    max_diff = np.abs(h_mine - h_torch).max()
    assert np.allclose(h_mine, h_torch, atol=1e-6), max_diff
    print(f"[PyTorch RNNCell] max |diff| = {max_diff:.2e}")

    # torch.nn.RNN 
    #xs = rng.normal(size=(batch, T, input_size))
    xs = rng.normal(size=(batch, T, input_size)).astype(np.float32)
    t_rnn = torch.nn.RNN(input_size, hidden_size, nonlinearity="tanh", batch_first=True)
    with torch.no_grad():
        t_rnn.weight_ih_l0.copy_(torch.from_numpy(cell.W_xh))
        t_rnn.weight_hh_l0.copy_(torch.from_numpy(cell.W_hh))
        t_rnn.bias_ih_l0.copy_(torch.from_numpy(cell.b_h))
        t_rnn.bias_hh_l0.zero_()
        out_torch, hT_torch = t_rnn(torch.from_numpy(xs))

    hs_mine, hT_mine = cell.forward_sequence(xs)
    max_diff_seq = np.abs(hs_mine - out_torch.numpy()).max()
    assert np.allclose(hs_mine, out_torch.numpy(), atol=1e-6), max_diff_seq
    assert np.allclose(hT_mine, hT_torch.numpy()[0], atol=1e-6)
    print(f"[PyTorch RNN]     max |diff| = {max_diff_seq:.2e}  (T={T}, batch={batch})")


def test_small_input():
    cell = SimpleRNNCell(input_size=3, hidden_size=2, seed=42)
    xs = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]])
    hs, h_T = cell.forward_sequence(xs)
    assert hs.shape == (3, 2) and h_T.shape == (2,)
    assert np.all(np.abs(hs) < 1.0)  # tanh -> mọi thành phần nằm trong (-1, 1)
    print("[Small input] hidden states qua 3 timestep:")
    print(np.round(hs, 4))


if __name__ == "__main__":
    test_bai1()
    test_small_input()
    compare_with_pytorch()
    print("SimpleRNNCell hoạt động đúng.")