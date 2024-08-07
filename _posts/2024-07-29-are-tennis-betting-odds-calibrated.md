---
layout: post
title: Determining No-Vig Fair Odds
date: 2024-07-29
---

Sportsbooks have seen a massive rise in the United States over the last few years.  Major sportsbooks such as FanDuel, Draft Kings, and Prize Picks.


### Mathematical set up
Consider an event with $$n \in \mathbb{N}$$ possible outcomes and let $$p_k = P(E_k)$$ where $$E_1,\ldots,E_n$$ denote the mutually exclusive outcomes.  Let $$W_k$$ denote the total winnings earned (original dollar bet plus additional winnings) whenever event $$E_k$$ occurs.  The payoff for betting $1 on outcome $$k$$ is then the random variable

$$
U(k) = \begin{cases}
W_k - 1 & \text{w.p.} & p_k\\
-1 & \text{w.p.} & 1 - p_k
\end{cases}
$$

Under "fair-odds", the expected payoff is $0 by definition or a zero-sum game.  In other words, if there were a game with 2 outcomes, then betting $1 on each outcome should simply result in the bettor recuperating their money with no additional winnings.  

For many sportsbooks the winnings may be quoted in "decimal" or "European" odds.  For example, a payout may be written as $$W = 1.16$$ indicating that winning the bet will result in a net profit of $$1.16 - 1 = 0.16$$ dollars per dollar bet.  Under this convention $$W > 1$$ always because you would never bet $1 if you were guaranted to earn less than that.

Let $$q_k$$ denote the fair implied probability based on the bookmaker's odds.  Under the zero-sum payoff condition we have that

$$
\mathbb{E}_{q_k}[U(k)] = (W_k - 1)q_k - (1 - q_k) = 0
$$

Solving for $$q_k$$ results in $$q_k = 1/W_k$$.


## Calculating implied probability from betting odds

### Decimal odds

### American odds


## Scraping
