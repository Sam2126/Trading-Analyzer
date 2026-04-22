"""
PrimeTrade.ai Hiring Assignment
Bitcoin Sentiment vs Trader Performance Analysis
Dataset: Hyperliquid Historical Trades + Bitcoin Fear/Greed Index
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────

trades_df = pd.read_csv('historical_data.csv')
sentiment_df = pd.read_csv('fear_greed_index.csv')

print("=== Dataset Overview ===")
print(f"Trades shape: {trades_df.shape}")
print(f"Sentiment shape: {sentiment_df.shape}")
print("\nTrades columns:", trades_df.columns.tolist())
print("Sentiment columns:", sentiment_df.columns.tolist())

# ─────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────

# Parse dates — trades use 'Timestamp IST' in format DD-MM-YYYY HH:MM (dayfirst)
trades_df['date'] = pd.to_datetime(trades_df['Timestamp IST'], dayfirst=True).dt.date

# Sentiment CSV uses 'date' (YYYY-MM-DD) and 'classification' (lowercase)
sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
sentiment_df['Classification'] = sentiment_df['classification'].str.strip().str.title()

# Clean PnL — column is 'Closed PnL'
trades_df = trades_df.dropna(subset=['Closed PnL'])
trades_df['Closed PnL'] = pd.to_numeric(trades_df['Closed PnL'], errors='coerce')
trades_df = trades_df.dropna(subset=['Closed PnL'])

# Win/loss flag — only count non-zero PnL rows as actual closed trades
trades_df['is_win'] = trades_df['Closed PnL'] > 0

# Side is 'BUY'/'SELL', Direction has richer values (Open Long, Close Short, etc.)
# Normalise Side to Long/Short for analysis
trades_df['side_clean'] = trades_df['Side'].str.upper().map({'BUY': 'Long', 'SELL': 'Short'})

print("\n=== After Cleaning ===")
print(f"Total rows: {len(trades_df):,}")
print(f"Rows with non-zero PnL: {(trades_df['Closed PnL'] != 0).sum():,}")
print(f"\nSentiment distribution:\n{sentiment_df['Classification'].value_counts()}")

# ─────────────────────────────────────────
# 3. MERGE DATASETS
# ─────────────────────────────────────────

merged_df = trades_df.merge(
    sentiment_df[['date', 'Classification']],
    on='date', how='left'
)
merged_df = merged_df.dropna(subset=['Classification'])

# Focus on trades that actually closed (non-zero PnL) for PnL analysis
closed_df = merged_df[merged_df['Closed PnL'] != 0].copy()

print(f"\nMerged rows (all): {len(merged_df):,}")
print(f"Merged rows (closed/non-zero PnL): {len(closed_df):,}")

# ─────────────────────────────────────────
# 4. CORE ANALYSIS
# ─────────────────────────────────────────

sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']

# A. PnL by Sentiment (using closed trades only)
pnl_by_sentiment = closed_df.groupby('Classification').agg(
    avg_pnl=('Closed PnL', 'mean'),
    total_pnl=('Closed PnL', 'sum'),
    median_pnl=('Closed PnL', 'median'),
    trade_count=('Closed PnL', 'count'),
    win_rate=('is_win', 'mean')
).reindex(sentiment_order)

print("\n=== PnL by Sentiment ===")
print(pnl_by_sentiment.round(2))

# B. Side (Long/Short) analysis
side_sentiment = closed_df.groupby(['Classification', 'side_clean'])['Closed PnL'].mean().unstack()
print("\n=== Avg PnL by Side and Sentiment ===")
print(side_sentiment.round(2))

# C. Direction breakdown (Open Long / Close Short etc.)
dir_counts = closed_df.groupby(['Classification', 'Direction'])['Closed PnL'].agg(['mean', 'count'])
print("\n=== Avg PnL by Direction and Sentiment (top directions) ===")
print(dir_counts[dir_counts['count'] > 100].round(2))

# D. Top Coin analysis
coin_pnl = closed_df.groupby('Coin').agg(
    avg_pnl=('Closed PnL', 'mean'),
    total_pnl=('Closed PnL', 'sum'),
    trade_count=('Closed PnL', 'count'),
    win_rate=('is_win', 'mean')
).sort_values('total_pnl', ascending=False)
print("\n=== Top 10 Coins by Total PnL ===")
print(coin_pnl.head(10).round(2))

# E. Top/Bottom Trader Performance
account_perf = closed_df.groupby('Account').agg(
    total_pnl=('Closed PnL', 'sum'),
    win_rate=('is_win', 'mean'),
    trade_count=('Closed PnL', 'count')
).sort_values('total_pnl', ascending=False)

print("\n=== Top 10 Accounts by Total PnL ===")
print(account_perf.head(10).round(2))

print("\n=== Bottom 10 Accounts by Total PnL ===")
print(account_perf.tail(10).round(2))

# F. Sentiment Lag Effect (1-day lag)
sentiment_sorted = sentiment_df.sort_values('date').copy()
sentiment_sorted['lag1_class'] = sentiment_sorted['Classification'].shift(1)

merged_lag = trades_df.merge(
    sentiment_sorted[['date', 'lag1_class']],
    on='date', how='left'
)
merged_lag = merged_lag[merged_lag['Closed PnL'] != 0]

lag_pnl = merged_lag.groupby('lag1_class')['Closed PnL'].mean().reindex(sentiment_order)
lag_wr  = merged_lag.groupby('lag1_class')['is_win'].mean().reindex(sentiment_order) * 100

print("\n=== Avg PnL — 1-day lag after sentiment ===")
print(lag_pnl.round(2))
print("\n=== Win Rate % — 1-day lag after sentiment ===")
print(lag_wr.round(1))

# ─────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────

plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#0a0e1a')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

colors_sentiment = {
    'Extreme Fear': '#ef4444',
    'Fear':         '#f87171',
    'Neutral':      '#60a5fa',
    'Greed':        '#34d399',
    'Extreme Greed':'#10b981'
}
bar_colors = [colors_sentiment.get(s, '#60a5fa') for s in sentiment_order]

def style_ax(ax):
    ax.set_facecolor('#111827')
    ax.tick_params(colors='#64748b', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')

# ── Plot 1: Avg PnL by Sentiment ──
ax1 = fig.add_subplot(gs[0, 0:2])
style_ax(ax1)
vals = pnl_by_sentiment['avg_pnl'].fillna(0)
bars = ax1.bar(sentiment_order, vals, color=bar_colors, width=0.6, edgecolor='none')
ax1.axhline(0, color='#374151', linewidth=0.8)
ax1.set_title('Avg Closed PnL by Sentiment', color='#f1f5f9', fontsize=12, pad=10)
ax1.set_ylabel('Avg PnL ($)', color='#94a3b8', fontsize=9)
for bar, val in zip(bars, vals):
    offset = max(abs(val) * 0.04, 2)
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + (offset if val >= 0 else -offset * 4),
             f'${val:.1f}', ha='center', va='bottom', color='#e2e8f0', fontsize=8)

# ── Plot 2: Win Rate by Sentiment ──
ax2 = fig.add_subplot(gs[0, 2])
style_ax(ax2)
win_rates = pnl_by_sentiment['win_rate'].fillna(0) * 100
ax2.barh(sentiment_order, win_rates, color=bar_colors, edgecolor='none')
ax2.axvline(50, color='#374151', linewidth=0.8, linestyle='--')
ax2.set_title('Win Rate %', color='#f1f5f9', fontsize=12, pad=10)
for i, v in enumerate(win_rates):
    ax2.text(v + 0.5, i, f'{v:.1f}%', va='center', color='#e2e8f0', fontsize=8)

# ── Plot 3: Trade Volume by Sentiment ──
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3)
ax3.bar(sentiment_order, pnl_by_sentiment['trade_count'].fillna(0),
        color=bar_colors, edgecolor='none')
ax3.set_title('Trade Volume by Sentiment', color='#f1f5f9', fontsize=11, pad=10)
ax3.set_xticklabels(sentiment_order, rotation=30, ha='right')

# ── Plot 4: PnL Distribution box plot ──
ax4 = fig.add_subplot(gs[1, 1:])
style_ax(ax4)
# Clip extreme outliers for readability
data_by_sentiment = [
    np.clip(closed_df[closed_df['Classification'] == s]['Closed PnL'].dropna().values, -5000, 5000)
    for s in sentiment_order
]
bp = ax4.boxplot(data_by_sentiment, labels=sentiment_order, patch_artist=True,
                 medianprops=dict(color='white', linewidth=1.5),
                 whiskerprops=dict(color='#374151'), capprops=dict(color='#374151'),
                 flierprops=dict(marker='.', markersize=2, color='#374151'))
for patch, color in zip(bp['boxes'], bar_colors):
    patch.set_facecolor(color + '44')
    patch.set_edgecolor(color)
ax4.set_title('PnL Distribution by Sentiment (clipped ±$5k)', color='#f1f5f9', fontsize=11, pad=10)
ax4.set_ylabel('Closed PnL ($)', color='#94a3b8', fontsize=9)
ax4.set_xticklabels(sentiment_order, rotation=25, ha='right', fontsize=7)

# ── Plot 5: Monthly PnL heatmap ──
ax5 = fig.add_subplot(gs[2, :])
style_ax(ax5)
closed_df['month'] = pd.to_datetime(closed_df['date']).dt.to_period('M')
heatmap_data = closed_df.groupby(['month', 'Classification'])['Closed PnL'].mean().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(columns=sentiment_order, fill_value=0)
vmax = heatmap_data.abs().quantile(0.95).max()  # dynamic scale
im = ax5.imshow(heatmap_data.values.T, aspect='auto', cmap='RdYlGn',
                vmin=-vmax, vmax=vmax)
ax5.set_yticks(range(len(sentiment_order)))
ax5.set_yticklabels(sentiment_order, color='#94a3b8', fontsize=8)
ax5.set_xticks(range(len(heatmap_data.index)))
ax5.set_xticklabels([str(m) for m in heatmap_data.index],
                    rotation=45, ha='right', color='#64748b', fontsize=7)
ax5.set_title('Monthly Avg PnL Heatmap by Sentiment Class', color='#f1f5f9', fontsize=11, pad=10)
plt.colorbar(im, ax=ax5, label='Avg PnL ($)', shrink=0.5)

plt.suptitle('Bitcoin Sentiment × Trader Performance — PrimeTrade.ai Assignment',
             color='#f1f5f9', fontsize=15, y=1.01, fontweight='bold')

plt.savefig('trader_sentiment_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0e1a', edgecolor='none')
plt.show()
print("\nChart saved → trader_sentiment_analysis.png")

# ─────────────────────────────────────────
# 6. FINAL SUMMARY
# ─────────────────────────────────────────

print("\n" + "="*60)
print("FINAL INSIGHTS SUMMARY")
print("="*60)

valid = pnl_by_sentiment.dropna(subset=['avg_pnl'])

best_sentiment  = valid['avg_pnl'].idxmax()
worst_sentiment = valid['avg_pnl'].idxmin()
best_win_rate   = valid['win_rate'].idxmax()

print(f"\n1. Best performing sentiment zone : {best_sentiment}")
print(f"   Avg PnL  : ${pnl_by_sentiment.loc[best_sentiment, 'avg_pnl']:.2f}")

print(f"\n2. Worst performing sentiment zone: {worst_sentiment}")
print(f"   Avg PnL  : ${pnl_by_sentiment.loc[worst_sentiment, 'avg_pnl']:.2f}")

print(f"\n3. Highest win rate in            : {best_win_rate}")
print(f"   Win Rate : {pnl_by_sentiment.loc[best_win_rate, 'win_rate']*100:.1f}%")

print("\n4. Top traded coin by volume      :", coin_pnl['trade_count'].idxmax())
print("   Top coin by total PnL          :", coin_pnl['total_pnl'].idxmax())

print("\n5. Lag effect (1-day entry after sentiment shift):")
for s in sentiment_order:
    if pd.notna(lag_pnl.get(s)):
        print(f"   {s:<16} → Avg PnL ${lag_pnl[s]:.2f}  |  Win Rate {lag_wr[s]:.1f}%")

print("\n6. Strategic recommendations:")
print("   - GREED zones    : Long bias, trend-following, up to 10x leverage")
print("   - FEAR zones     : Cautious entry, max 3-5x leverage, await confirmation")
print("   - EXTREME GREED  : Begin shorting, reduce position size")
print("   - EXTREME FEAR   : Contrarian long — low leverage only")
print("   - 1-day lag rule : Enter the day AFTER a sentiment shift for better win rate")

print("\nAnalysis complete.")