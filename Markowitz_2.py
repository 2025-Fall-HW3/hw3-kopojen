"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=20, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below

        Implement a very aggressive short-term momentum with trend confirmation:
        1. Use 20-day lookback for ultra-responsive trading
        2. Select top 2 strongest trending assets
        3. Weight by momentum strength (not just equal or inv-vol)
        4. Add trend confirmation to avoid whipsaws
        """

        lookback = self.lookback
        top_n = 2  # Very concentrated: top 2 only
        
        for i in range(lookback + 1, len(self.price)):
            window_returns = self.returns[assets].iloc[i - lookback : i]
            
            # Multiple momentum signals
            cum_returns = (1 + window_returns).prod() - 1  # Total return
            recent_5d = window_returns.tail(5).sum() if len(window_returns) >= 5 else window_returns.sum()
            
            # Combined momentum score (emphasize recent performance)
            momentum_score = cum_returns * 0.6 + recent_5d * 0.4
            
            # Get top N with strongest momentum
            top_momentum = momentum_score.nlargest(top_n)
            
            if len(top_momentum) > 0 and top_momentum.min() > 0:  # Only if positive momentum
                # Weight proportional to momentum strength
                weights = pd.Series(0.0, index=assets)
                weights[top_momentum.index] = top_momentum / top_momentum.sum()
                self.portfolio_weights.loc[self.price.index[i], assets] = weights
            else:
                # If no strong momentum, use top 4 with inverse volatility
                volatility = window_returns.std()
                volatility = volatility.replace(0, volatility[volatility > 0].min())
                inv_vol = 1.0 / volatility
                top_inv_vol = inv_vol.nlargest(4)
                
                weights = pd.Series(0.0, index=assets)
                weights[top_inv_vol.index] = top_inv_vol / top_inv_vol.sum()
                self.portfolio_weights.loc[self.price.index[i], assets] = weights

        # ensure excluded asset has zero weight
        self.portfolio_weights[self.exclude] = 0

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
