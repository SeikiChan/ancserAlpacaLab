import logging
import pandas as pd
from ancser_quant.data.alpaca_adapter import AlpacaAdapter

logger = logging.getLogger("AncserExecution")

class OrderManagementSystem:
    """
    Execution Layer (Body).
    Takes Target Weights -> Generates Orders -> Submits to Alpaca.
    """
    def __init__(self):
        self.alpaca = AlpacaAdapter() # Use adapter for API access

    def generate_and_execute_orders(self, target_weights: dict) -> list:
        """
        1. Get Current Portfolio (Positions + Cash)
        2. Calculate Target Value per Asset
        3. Calculate Diff (Orders)
        4. Execute Orders (Sell first, then Buy)
        """
        # 0. Cancel Open Orders (Free up Buying Power & Shares)
        self.alpaca.cancel_all_orders()
        
        # 1. Get Account Info
        acct = self.alpaca.get_account()
        if not acct:
            logger.error("Failed to get account info. Aborting rebalance.")
            return []
            
        equity = float(acct.get('equity', 0.0))
        buying_power = float(acct.get('buying_power', 0.0))
        logger.info(f"Account Equity: ${equity:,.2f}, Buying Power: ${buying_power:,.2f}")
        
        # Get Current Positions
        positions = self.alpaca.get_positions()
        current_holdings = {p['Symbol']: float(p['Market Value']) for p in positions}
        current_qtys = {p['Symbol']: float(p['Qty']) for p in positions}
        
        logger.info(f"Current Holdings: {list(current_holdings.keys())}")
        
        # 2. Calculate Orders
        orders = []
        skipped_trades = {}
        
        # Determine all involved symbols (Current + Target)
        all_symbols = set(current_holdings.keys()) | set(target_weights.keys())
        
        for sym in all_symbols:
            current_val = current_holdings.get(sym, 0.0)
            target_pct = target_weights.get(sym, 0.0)
            target_val = equity * target_pct
            
            diff_val = target_val - current_val
            
            # Threshold logic: 
            # - SELL orders for stocks NOT in target (target_pct=0): Force full liquidation, no minimum
            # - BUY/ADJUST orders: Minimum $0.01 to avoid noise
            MIN_TRADE_SIZE = 0.01
            is_liquidation = (target_pct == 0.0 and current_val > 0.0)
            
            if abs(diff_val) < MIN_TRADE_SIZE and not is_liquidation:
                skipped_trades[sym] = f"${abs(diff_val):.7f} (below ${MIN_TRADE_SIZE} threshold)"
                continue
                
            # Estimate Price to calculate Qty
            # We need latest price. Adapter get_positions uses 'current_price'. 
            # If we don't hold it, we need to fetch a snapshot.
            price = 0.0
            
            # Check if we have it in positions
            found = False
            for p in positions:
                if p['Symbol'] == sym:
                    price = float(p['Current Price'])
                    found = True
                    break
            
            if not found:
                # Need to fetch price
                # We can use alpaca-py trading client get_latest_trade or snapshot if available in adapter
                # Adapter doesn't expose snapshot explicitly, but we can try to get it or just use cached price from Preview if passed
                # For robustness, let's try a simple get_latest_trade via adapter's trading_client if possible, 
                # or assumes 0 and skip (bad).
                # Main Loop has 'latest_prices' from Strategy! We should pass that in.
                # For now, let's assume we can get it or use a default.
                try:
                    # Quick hack: use adapter's trading_client internal
                    # Or better: Adapter should have get_latest_price(sym)
                    # Let's rely on Main Loop passing prices? 
                    # If we change signature of this method to accept prices, it's cleaner.
                    # But to keep it valid based on task description, let's just make a single call.
                    # Actually, let's blindly calculate Notional Order? Alpaca supports Notional Orders!
                    pass
                except:
                    pass

            # Alpaca supports Notional (dollar amount) orders for most assets!
            # We can just submit "Buy $500 of AAPL" or "Sell $200 of MSFT".
            # For SELL orders, apply 0.01% safety discount to avoid precision loss issues
            
            side = 'buy' if diff_val > 0 else 'sell'
            qty_val = abs(diff_val)
            
            # Apply precision safety discount for SELL orders (Alpaca notional precision issue)
            if side == 'sell':
                qty_val = qty_val * 0.9999  # 0.01% safety margin
            
            orders.append({
                'symbol': sym,
                'side': side,
                'notional': qty_val,
                'type': 'market',  # Market order for now
                'is_liquidation': is_liquidation  # Flag for forced liquidation
            })

        # 3. Execution (Sell First, Then Buy)
        # Sell orders first to free up cash
        sell_orders = [o for o in orders if o['side'] == 'sell']
        buy_orders = [o for o in orders if o['side'] == 'buy']
        
        executed_orders = []
        failed_orders = []
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"ORDER EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Sell Orders: {len(sell_orders)}")
        logger.info(f"Total Buy Orders: {len(buy_orders)}")
        logger.info(f"Skipped Trades:   {len(skipped_trades)}")
        
        if skipped_trades:
            logger.info("")
            logger.info("Skipped Trades (Too Small):")
            for sym, reason in skipped_trades.items():
                logger.info(f"  - {sym}: {reason}")
        logger.info("")
        
        # Execute Sells
        for order in sell_orders:
            try:
                liquidation_tag = " [LIQUIDATE]" if order.get('is_liquidation') else ""
                logger.info(f"[SELL] {order['symbol']:6} ${order['notional']:>10.2f}{liquidation_tag}")
                self.alpaca.submit_order(
                    symbol=order['symbol'],
                    qty=0, # ignored if notional
                    side='sell',
                    notional=order['notional']
                )
                executed_orders.append(order)
                logger.info(f"       ✓ SUCCESS")
            except Exception as e:
                failed_orders.append((order, str(e)))
                logger.error(f"       ✗ FAILED: {e}")

        # Execute Buys
        for order in buy_orders:
            try:
                # Check buying power? Alpaca handles it.
                logger.info(f"[BUY]  {order['symbol']:6} ${order['notional']:>10.2f}")
                self.alpaca.submit_order(
                    symbol=order['symbol'],
                    qty=0, 
                    side='buy',
                    notional=order['notional']
                )
                executed_orders.append(order)
                logger.info(f"       ✓ SUCCESS")
            except Exception as e:
                failed_orders.append((order, str(e)))
                logger.error(f"       ✗ FAILED: {e}")
        
        # Final Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"EXECUTION RESULT:")
        logger.info(f"  Orders Submitted: {len(executed_orders)}")
        logger.info(f"  Orders Failed:    {len(failed_orders)}")
        logger.info("=" * 60)
        
        if len(executed_orders) == 0:
            logger.warning("⚠ NO ORDERS EXECUTED IN THIS REBALANCE!")
        else:
            logger.info(f"✓ {len(executed_orders)} orders submitted successfully")
        logger.info("")
        
        return executed_orders
