from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple
import json
from statistics import mean, stdev, median
import math

class MarketAnalyzer:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        
    def analyze_order_book(self, buy_orders: Dict[int, int], sell_orders: Dict[int, int]) -> Tuple[float, float, float]:
        """Analyze order book for imbalances and potential profit opportunities"""
        if not buy_orders or not sell_orders:
            return 0, 0, 0
            
        buy_volume = sum(abs(qty) for qty in buy_orders.values())
        sell_volume = sum(abs(qty) for qty in sell_orders.values())
        imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        
        vwap_buy = sum(price * abs(qty) for price, qty in buy_orders.items()) / buy_volume
        vwap_sell = sum(price * abs(qty) for price, qty in sell_orders.items()) / sell_volume
        
        return vwap_buy, vwap_sell, imbalance
        
    def update_history(self, product: str, mid_price: float, volume: float):
        """Update price and volume history"""
        if product not in self.price_history:
            self.price_history[product] = []
            self.volume_history[product] = []
            
        self.price_history[product].append(mid_price)
        self.volume_history[product].append(volume)
        
        if len(self.price_history[product]) > self.window_size:
            self.price_history[product] = self.price_history[product][-self.window_size:]
            self.volume_history[product] = self.volume_history[product][-self.window_size:]
            
    def get_volatility(self, product: str) -> float:
        """Calculate price volatility"""
        if product not in self.price_history or len(self.price_history[product]) < 2:
            return 1.0
        return stdev(self.price_history[product])

class Trader:
    def __init__(self):
        self.position_limits = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50
        }
        self.analyzer = MarketAnalyzer()
        self.min_spread = {  
            "KELP": 2,
            "RAINFOREST_RESIN": 2
        }
        
    def calculate_dynamic_spread(self, product: str, volatility: float, imbalance: float) -> float:
        """Calculate dynamic spread based on volatility and order book imbalance"""
        base_spread = self.min_spread.get(product, 2)
        volatility_factor = volatility * 0.5
        imbalance_factor = abs(imbalance) * 2
        return max(base_spread, base_spread + volatility_factor + imbalance_factor)
        
    def calculate_order_size(self, product: str, current_pos: int, pos_limit: int, imbalance: float) -> tuple[int, int]:
        """Calculate aggressive order sizes based on market imbalance"""
        base_size = int(pos_limit * 0.2)
        
        if imbalance > 0:  
            buy_size = int(base_size * (1 - imbalance))
            sell_size = int(base_size * (1 + imbalance))
        else:  
            buy_size = int(base_size * (1 - imbalance))
            sell_size = int(base_size * (1 + imbalance))
            
        position_ratio = abs(current_pos) / pos_limit if pos_limit > 0 else 1
        if current_pos > 0:
            sell_size = int(sell_size * (1 + position_ratio)) 
        elif current_pos < 0:
            buy_size = int(buy_size * (1 + position_ratio))  
            
        buy_size = min(max(1, buy_size), pos_limit - current_pos)
        sell_size = min(max(1, sell_size), pos_limit + current_pos)
        
        return buy_size, sell_size
        
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
            
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue

            vwap_buy, vwap_sell, imbalance = self.analyzer.analyze_order_book(
                order_depth.buy_orders, order_depth.sell_orders
            )
            
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            
            total_volume = sum(abs(qty) for qty in order_depth.buy_orders.values()) + \
                          sum(abs(qty) for qty in order_depth.sell_orders.values())
            self.analyzer.update_history(product, mid_price, total_volume)
            
            volatility = self.analyzer.get_volatility(product)
            spread = self.calculate_dynamic_spread(product, volatility, imbalance)
            
            if imbalance > 0.3:  
                our_bid = best_bid  
                our_ask = int(best_ask + spread) 
            elif imbalance < -0.3:  
                our_bid = int(best_bid - spread)  
                our_ask = best_ask  
            else:  
                our_bid = int(mid_price - spread/2)
                our_ask = int(mid_price + spread/2)

            current_pos = state.position.get(product, 0)
            pos_limit = self.position_limits.get(product, 50)
            
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
            
            buy_size, sell_size = self.calculate_order_size(product, current_pos, pos_limit, imbalance)
            
            if buy_size > 0 and current_pos < pos_limit:
                if our_bid < best_ask:  
                    print("BUY " + str(buy_size) + "x " + str(our_bid))
                    orders.append(Order(product, our_bid, buy_size))
                    
            if sell_size > 0 and current_pos > -pos_limit:
                if our_ask > best_bid:  
                    print("SELL " + str(sell_size) + "x " + str(our_ask))
                    orders.append(Order(product, our_ask, -sell_size))
            
            result[product] = orders

        traderData = json.dumps({
            "last_timestamp": state.timestamp,
            "positions": state.position
        })
        
        return result, 1, traderData