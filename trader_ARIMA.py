from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple
import json
from statistics import mean, stdev, median
import math

import numpy as np


class ARIMA:
    def __init__(self, p=1, d=1, q=1, learning_rate=0.01, epochs=100):
        self.p = p
        self.d = d
        self.q = q
        self.lr = learning_rate
        self.epochs = epochs
        self.params = None
        self.history = None
    
    def difference(self, data):
        """执行d阶差分，返回差分后的数据"""
        diff = data.copy()
        for _ in range(self.d):
            diff = np.diff(diff)
        return diff
    
    def inverse_difference(self, last_obs, diff_pred):
        """将差分预测值逆转换到原始数据空间"""
        return last_obs + np.cumsum(diff_pred)
    
    def compute_loss(self, y, params):
        """计算预测误差的均方损失"""
        c, ar, ma = params[0], params[1:1 + self.p], params[1 + self.p:]
        n = len(y)
        max_lag = max(self.p, self.q)
        epsilon = np.zeros(n)
        loss = 0.0
        
        for t in range(max_lag, n):
            ar_term = np.dot(ar, y[t - self.p:t][::-1])
            ma_term = np.dot(ma, epsilon[t - self.q:t][::-1])
            y_pred = c + ar_term + ma_term
            epsilon[t] = y[t] - y_pred
            loss += epsilon[t] ** 2
        
        return loss / (n - max_lag)
    
    def fit(self, data):
        y = self.difference(data)
        self.history = y

        self.params = np.zeros(1 + self.p + self.q) + 0.01  # 添加小扰动避免零梯度
        
        for epoch in range(self.epochs):
            loss = self.compute_loss(y, self.params)
            grad = np.zeros_like(self.params)
            h = 1e-6
            
            for i in range(len(self.params)):
                params_temp = self.params.copy()
                params_temp[i] += h
                loss_high = self.compute_loss(y, params_temp)
                params_temp[i] -= 2 * h
                loss_low = self.compute_loss(y, params_temp)
                grad[i] = (loss_high - loss_low) / (2 * h)
            
            self.params -= self.lr * grad
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    
    def forecast(self, steps=1,data=None):
        y_diff = self.history.copy()
        n = len(y_diff)
        max_lag = max(self.p, self.q)
        c, ar, ma = self.params[0], self.params[1:1 + self.p], self.params[1 + self.p:]
        epsilon = np.zeros(n + steps)
        
        forecasts = []
        for _ in range(steps):
            ar_vals = y_diff[-self.p:][::-1] if len(y_diff) >= self.p else np.zeros(self.p)
            ar_term = np.dot(ar, ar_vals)
            
            ma_vals = epsilon[n - self.q: n][::-1] if n >= self.q else np.zeros(self.q)
            ma_term = np.dot(ma, ma_vals)
            
            y_pred = c + ar_term + ma_term
            forecasts.append(y_pred)
            
            y_diff = np.append(y_diff, y_pred)
            epsilon[n] = y_pred - y_pred
            n += 1
        
        last_original = data[-self.d - 1] if len(data) >= self.d + 1 else data[0]
        forecast_diff = np.array(forecasts)
        
        return self.inverse_difference(last_original, forecast_diff)

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

# 拥有足够数据后在使用此方法预测,至少每个产品高于5个数据点
def forcast_by_ARIMA(MA: MarketAnalyzer,step):
    model_table = ARIMA(p=1,q=1,d=0)
    model_high_ar = ARIMA(p=3,q=0,d=1)
    model_low_ar = ARIMA(p=0,q=2,d=1)
    prices_res = MA.price_history.copy()
    volumes_res = MA.volume_history.copy()
    for product in prices_res.keys():
        model_table.fit(np.array(prices_res[product])),
        model_high_ar.fit(np.array(prices_res[product])),
        model_low_ar.fit(np.array(prices_res[product])),
        res = np.asarray([
            model_table.forecast(step,prices_res[product]),
            model_high_ar.forecast(step,prices_res[product]),
            model_low_ar.forecast(step,prices_res[product])
        ])
        prices_res[product] = res.mean(axis=0).tolist()
    
    for product in volumes_res.keys():
        model_table.fit(np.array(volumes_res[product],volumes_res[product])),
        model_high_ar.fit(np.array(volumes_res[product],volumes_res[product])),
        model_low_ar.fit(np.array(volumes_res[product],volumes_res[product])),
        res = np.asarray([
            model_table.forecast(step),
            model_high_ar.forecast(step),
            model_low_ar.forecast(step)
        ])
        volumes_res[product] = res.mean(axis=0).tolist()
    
    return prices_res, volumes_res

class ARIMALogger:
    def __init__(self):
        self.activities = []
        
    def log_prediction(self, timestamp: int, product: str, predictions: Dict[str, List[float]], actual_price: float):
        """Log ARIMA predictions and actual price"""
        activity = {
            'timestamp': timestamp,
            'product': product,
            'predictions': {
                'table_model': predictions['table'][0] if len(predictions['table']) > 0 else None,
                'high_ar_model': predictions['high_ar'][0] if len(predictions['high_ar']) > 0 else None,
                'low_ar_model': predictions['low_ar'][0] if len(predictions['low_ar']) > 0 else None,
                'ensemble_prediction': np.mean([
                    predictions['table'][0] if len(predictions['table']) > 0 else 0,
                    predictions['high_ar'][0] if len(predictions['high_ar']) > 0 else 0,
                    predictions['low_ar'][0] if len(predictions['low_ar']) > 0 else 0
                ])
            },
            'actual_price': actual_price,
            'prediction_error': None  # Will be updated in next timestamp
        }
        self.activities.append(activity)
        
        # Update previous prediction error if exists
        if len(self.activities) > 1:
            prev_activity = self.activities[-2]
            prev_activity['prediction_error'] = actual_price - prev_activity['predictions']['ensemble_prediction']
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics for predictions"""
        if not self.activities:
            return {}
            
        errors = [act['prediction_error'] for act in self.activities if act['prediction_error'] is not None]
        if not errors:
            return {}
            
        return {
            'mean_error': np.mean(errors),
            'mean_absolute_error': np.mean(np.abs(errors)),
            'root_mean_squared_error': np.sqrt(np.mean(np.array(errors) ** 2)),
            'prediction_count': len(errors)
        }
    
    def format_activity_log(self) -> str:
        """Format activity log for display"""
        output = []
        output.append("\n=== ARIMA Trading Activity Log ===")
        
        for activity in self.activities:
            output.append(f"\nTimestamp: {activity['timestamp']}")
            output.append(f"Product: {activity['product']}")
            output.append("Predictions:")
            output.append(f"  Table Model: {activity['predictions']['table_model']:.2f}")
            output.append(f"  High AR Model: {activity['predictions']['high_ar_model']:.2f}")
            output.append(f"  Low AR Model: {activity['predictions']['low_ar_model']:.2f}")
            output.append(f"  Ensemble: {activity['predictions']['ensemble_prediction']:.2f}")
            output.append(f"Actual Price: {activity['actual_price']:.2f}")
            if activity['prediction_error'] is not None:
                output.append(f"Prediction Error: {activity['prediction_error']:.2f}")
            output.append("-" * 50)
        
        metrics = self.get_performance_metrics()
        if metrics:
            output.append("\nPerformance Metrics:")
            output.append(f"Mean Error: {metrics['mean_error']:.2f}")
            output.append(f"Mean Absolute Error: {metrics['mean_absolute_error']:.2f}")
            output.append(f"Root Mean Squared Error: {metrics['root_mean_squared_error']:.2f}")
            output.append(f"Total Predictions: {metrics['prediction_count']}")
        
        return "\n".join(output)

class AssetTracker:
    def __init__(self):
        self.cash = 0.0
        self.positions = {}  # {product: quantity}
        self.position_values = {}  # {product: current_value}
        self.history = []  # [{timestamp, cash, positions, total_value}, ...]
        
    def update(self, timestamp: int, positions: Dict[str, int], mid_prices: Dict[str, float]):
        """Update asset tracker with new positions and prices"""
        self.positions = positions.copy()
        
        # Calculate position values
        total_position_value = 0
        for product, quantity in positions.items():
            if product in mid_prices:
                value = quantity * mid_prices[product]
                self.position_values[product] = value
                total_position_value += value
        
        # Record history
        snapshot = {
            'timestamp': timestamp,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'position_values': self.position_values.copy(),
            'total_value': self.cash + total_position_value
        }
        self.history.append(snapshot)
    
    def update_cash(self, trade_price: float, trade_quantity: int):
        """Update cash based on trade"""
        self.cash -= trade_price * trade_quantity  # Buy reduces cash, sell increases cash
    
    def get_current_value(self) -> float:
        """Get total portfolio value"""
        if not self.history:
            return self.cash
        return self.history[-1]['total_value']
    
    def get_position_value(self, product: str) -> float:
        """Get current value of a specific position"""
        return self.position_values.get(product, 0.0)
    
    def format_status(self) -> str:
        """Format current status for display"""
        output = []
        output.append("\n=== Asset Status ===")
        output.append(f"Cash: {self.cash:.2f}")
        
        for product, quantity in self.positions.items():
            value = self.position_values.get(product, 0.0)
            output.append(f"{product}:")
            output.append(f"  Quantity: {quantity}")
            output.append(f"  Value: {value:.2f}")
        
        output.append(f"Total Portfolio Value: {self.get_current_value():.2f}")
        return "\n".join(output)

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
        self.arima_logger = ARIMALogger()
        self.asset_tracker = AssetTracker()  # Add asset tracker
        
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
        # Track mid prices for asset value calculation
        current_mid_prices = {}
        
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
            current_mid_prices[product] = mid_price
            
            total_volume = sum(abs(qty) for qty in order_depth.buy_orders.values()) + \
                          sum(abs(qty) for qty in order_depth.sell_orders.values())
            self.analyzer.update_history(product, mid_price, total_volume)
            
            # If we have enough data points, make ARIMA predictions
            if len(self.analyzer.price_history[product]) >= 5:
                prices_pred, volumes_pred = forcast_by_ARIMA(self.analyzer, 1)
                
                # Log predictions
                self.arima_logger.log_prediction(
                    timestamp=state.timestamp,
                    product=product,
                    predictions={
                        'table': prices_pred[product],
                        'high_ar': prices_pred[product],
                        'low_ar': prices_pred[product]
                    },
                    actual_price=mid_price
                )
                
                # Print activity log
                print(self.arima_logger.format_activity_log())
            
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
        
        # Update asset tracker
        self.asset_tracker.update(state.timestamp, state.position, current_mid_prices)
        
        # Update cash based on new trades
        for product, orders in result.items():
            for order in orders:
                self.asset_tracker.update_cash(order.price, order.quantity)
        
        # Print asset status
        print(self.asset_tracker.format_status())

        traderData = json.dumps({
            "last_timestamp": state.timestamp,
            "positions": state.position,
            "arima_metrics": self.arima_logger.get_performance_metrics(),
            "portfolio_value": self.asset_tracker.get_current_value()
        })
        
        return result, 1, traderData

class LogParser:
    @staticmethod
    def parse_log_line(log_line: str) -> Dict:
        """Parse a single log line into structured data"""
        # Example log line: -1;0;RAINFOREST_RESIN;10002;1;9996;2;9995;29;10004;2;10005;29;;;10003.0;0.0
        parts = log_line.split(';')
        
        return {
            'day': int(parts[0]),
            'timestamp': int(parts[1]),
            'product': parts[2],
            'bids': [
                {'price': int(parts[3]), 'volume': int(parts[4])},
                {'price': int(parts[5]), 'volume': int(parts[6])},
                {'price': int(parts[7]), 'volume': int(parts[8])}
            ],
            'asks': [
                {'price': int(parts[9]), 'volume': int(parts[10])},
                {'price': int(parts[11]), 'volume': int(parts[12])},
                {'price': int(parts[13]) if parts[13] else None, 'volume': int(parts[14]) if parts[14] else None}
            ],
            'mid_price': float(parts[15]),
            'profit_and_loss': float(parts[16])
        }
    
    @staticmethod
    def parse_log_file(log_file_path: str) -> List[Dict]:
        """Parse entire log file into structured data"""
        parsed_data = []
        with open(log_file_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    parsed_data.append(LogParser.parse_log_line(line.strip()))
        return parsed_data

    @staticmethod
    def to_market_analyzer(log_data: List[Dict], window_size: int = 10) -> MarketAnalyzer:
        """Convert parsed log data into MarketAnalyzer format"""
        ma = MarketAnalyzer(window_size=window_size)
        product_data = {}
        for entry in log_data:
            product = entry['product']
            if product not in product_data:
                product_data[product] = []
            product_data[product].append(entry)
        for product, entries in product_data.items():
            entries.sort(key=lambda x: x['timestamp'])
            for entry in entries:
                mid_price = entry['mid_price']

                total_volume = sum(bid['volume'] for bid in entry['bids'] if bid['volume'] is not None)
                total_volume += sum(ask['volume'] for ask in entry['asks'] if ask['volume'] is not None)
                ma.update_history(product, mid_price, total_volume) 
        return ma

    @staticmethod
    def create_order_depth_from_log(log_entry: Dict) -> OrderDepth:
        """Create OrderDepth object from log entry"""
        order_depth = OrderDepth()

        for bid in log_entry['bids']:
            if bid['price'] is not None and bid['volume'] is not None:
                order_depth.buy_orders[bid['price']] = bid['volume']

        for ask in log_entry['asks']:
            if ask['price'] is not None and ask['volume'] is not None:
                order_depth.sell_orders[ask['price']] = ask['volume']
        
        return order_depth

# Example usage:
"""
# Read log file and convert to MarketAnalyzer
log_data = LogParser.parse_log_file('trading_log.txt')
market_analyzer = LogParser.to_market_analyzer(log_data)

# Access historical data
print(market_analyzer.price_history)
print(market_analyzer.volume_history)

# Get latest order depth for a product
latest_entry = log_data[-1]
order_depth = LogParser.create_order_depth_from_log(latest_entry)
"""