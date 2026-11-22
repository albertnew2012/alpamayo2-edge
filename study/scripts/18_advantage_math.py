"""How `loss = -advantage * log p(plan)` raises or lowers a plan's probability (CPU, instant).

    python study/scripts/18_advantage_math.py

1. One plan, three advantages: +1 raises p(plan), 0 leaves it, -1 lowers it.
2. A whole group (the 8 plans of 17_rl_toy.py section 3): every plan pulls the mean toward
   itself (A > 0) or pushes it away (A < 0), weighted by its advantage.
3. Any model, not just a Gaussian: one gradient step changes log p(plan) by about
   lr * A * |grad log p|^2, whose sign is the sign of A.
"""

import torch

torch.manual_seed(0)
SIGMA, LR = 0.2, 0.01


def log_p(mu, y):
    """log N(y; mu, SIGMA^2) = -(y - mu)^2 / (2 SIGMA^2) - log SIGMA - 0.5 log(2 pi)."""
    return torch.distributions.Normal(mu, SIGMA).log_prob(y)


print("1. One plan y = 1.90 m, the planner's mean mu = 1.70 m, one step on loss = -A * log p(y)")
y = torch.tensor(1.90)
for A in (+1.0, 0.0, -1.0):
    mu = torch.tensor(1.70, requires_grad=True)
    before = log_p(mu, y).exp().item()
    loss = -A * log_p(mu, y)
    loss.backward()
    formula = -A * (y - 1.70) / SIGMA**2  # d loss / d mu = -A * (y - mu) / sigma^2
    with torch.no_grad():
        mu -= LR * mu.grad  # gradient descent
    print(f"   A = {A:+.0f}:  d loss/d mu = {mu.grad.item():+6.2f} (formula {formula.item():+6.2f})"
          f"   mu 1.700 -> {mu.item():.3f}   p(y) {before:.3f} -> {log_p(mu, y).exp().item():.3f}")

print("\n2. A group of 8 plans for one scene (collision below 1.40 m)")
plans = torch.tensor([1.30, 1.34, 1.37, 1.66, 1.75, 1.79, 1.85, 1.94])
scores = torch.tensor([0.00, 0.00, 0.00, 0.72, 0.79, 0.82, 0.87, 0.95])
adv = (scores - scores.mean()) / scores.std()
mu = torch.tensor(1.70, requires_grad=True)
p_before = log_p(mu, plans).exp().detach()
loss = -(adv * log_p(mu, plans)).mean()
loss.backward()
pull = adv * (plans - 1.70) / SIGMA**2 / len(plans)  # each plan's share of -d loss / d mu
with torch.no_grad():
    mu -= LR * mu.grad
p_after = log_p(mu, plans).exp()
print("   plan     y   score   advantage   pull on mu   p(plan) before -> after")
for k in range(len(plans)):
    print(f"   {k + 1:4d}  {plans[k]:.2f}   {scores[k]:.2f}    {adv[k]:+6.2f}      {pull[k]:+6.3f}"
          f"        {p_before[k]:.3f} -> {p_after[k]:.3f}  {'up' if p_after[k] > p_before[k] else 'down'}")
print(f"   total pull {pull.sum():+.3f} = -d loss/d mu {-mu.grad.item():+.3f}"
      f"   ->  mu 1.700 -> {mu.item():.3f}")
print("   Crashed plans lie below mu and have A < 0: (y - mu) < 0 times A < 0 = a pull UP, away")
print("   from them. Good plans lie above mu and have A > 0: also a pull up, toward them.")
print("   Plan 4 has A > 0, so its OWN pull is toward it (-0.058, slightly down), but the other plans")
print("   pull harder the other way: the net step moves mu past it and p(plan 4) drops. Each plan's")
print("   term pushes its own probability in the direction of its advantage; the net change is the")
print("   sum over the group, so a 'good but not best' plan can still lose probability.")

print("\n3. Any model: a tiny MLP maps the scene feature c to mu; check the first-order rule")
net = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
with torch.no_grad():
    net[2].bias += 1.70 - net(torch.tensor([[0.4]])).squeeze()  # start with mu = 1.70 for c = 0.4
c, y = torch.tensor([[0.4]]), torch.tensor([[1.90]])
STEP = 1e-3  # small step: the rule is first order
for A in (+1.0, -1.0):
    model = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    model.load_state_dict(net.state_dict())
    lp = log_p(model(c), y).sum()
    grads = torch.autograd.grad(lp, list(model.parameters()))
    sq_norm = sum((g**2).sum() for g in grads)          # |grad_theta log p(y)|^2
    with torch.no_grad():
        for w, g in zip(model.parameters(), grads, strict=True):
            w += STEP * A * g                            # = gradient descent on -A * log p
        change = log_p(model(c), y).sum() - lp
    print(f"   A = {A:+.0f}:  change in log p(y) {change.item():+.5f}   "
          f"predicted lr*A*|grad|^2 = {STEP * A * sq_norm.item():+.5f}")
print("   |grad|^2 is never negative, so the sign of the change is the sign of A, for any model.")
