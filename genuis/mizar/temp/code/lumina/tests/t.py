from ultron.kdutils.progress import Progress
from ultron.utilities.logger import kd_logger
from ultron.kdutils.date import str_to_datetime
import pandas as pd
import matplotlib.pyplot as plt
# import pdb # Removed pdb as it's not used in the final version

def plot_his_trade(orders,
                   kl_pd,
                   y_zoon=1.5,
                   fmt='%Y-%m-%d %H:%M:%S',
                   alignment=False): # alignment parameter seems unused, consider removing if not needed
    """
    可视化绘制Order对象，绘制交易买入时间，卖出时间，价格等
    :param orders: Order对象序列
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :return:
    """
    # Ensure kl_pd index is datetime for proper time-based operations
    if not isinstance(kl_pd.index, pd.DatetimeIndex):
         try:
             # Assuming the index is string representable time
             kl_pd.index = pd.to_datetime(kl_pd.index)
         except Exception as e:
             kd_logger.error(f"Failed to convert kl_pd index to DatetimeIndex: {e}")
             # Potentially fallback or raise error if conversion fails
             # If 'trade_time' column exists and index isn't time, maybe use that?
             if 'trade_time' in kl_pd.columns:
                 try:
                     kl_pd.index = pd.to_datetime(kl_pd['trade_time'])
                     kd_logger.info("Using 'trade_time' column as index.")
                 except Exception as e_inner:
                     kd_logger.error(f"Failed to convert 'trade_time' to DatetimeIndex: {e_inner}")
                     return # Cannot proceed without a time index
             else:
                 return # Cannot proceed

    # Check if 'key' column exists, if not, create it
    if 'key' not in kl_pd.columns:
        kl_pd['key'] = range(len(kl_pd))
        kd_logger.info("Added 'key' column for indexing.")

    # Check if 'trade_time' column exists for matching, if not, use index
    if 'trade_time' not in kl_pd.columns:
        # Assuming index is the correct trade time after conversion above
        kl_pd['trade_time'] = kl_pd.index
        kd_logger.info("Using index as 'trade_time' column.")
    else:
        # Ensure 'trade_time' is in datetime format if it exists
        try:
            kl_pd['trade_time'] = pd.to_datetime(kl_pd['trade_time'])
        except Exception as e:
             kd_logger.warning(f"Could not convert existing 'trade_time' column to datetime: {e}")
             # Consider if this is critical or if string comparison might work (less robust)


    # 拿出时间序列中最后一个，做为当前价格
    if kl_pd.empty:
        kd_logger.error("Input kl_pd DataFrame is empty.")
        return
    now_price = kl_pd.iloc[-1].close
    all_pd = kl_pd

    ax_cnt = 1.5 * len(orders)
    fig, ax = plt.subplots(figsize=(14 + ax_cnt, 8 * y_zoon))
    # Use index directly for plotting if it's DatetimeIndex
    times_plot = all_pd.index # Use datetime objects for plotting

    ## --- START: MODIFICATION 1 ---
    # Calculate min/max and padding *before* plotting annotations
    min_close = all_pd['close'].min()
    max_close = all_pd['close'].max()
    padding = (max_close - min_close) * 0.1 # Original padding
    # Add slightly more padding specifically for annotations if needed
    annotation_padding_factor = 0.05 # Add 5% extra range for annotations
    ylim_min = min_close - padding - (max_close - min_close) * annotation_padding_factor
    ylim_max = max_close + padding + (max_close - min_close) * annotation_padding_factor
    y_range = ylim_max - ylim_min # Calculate the full visible y-range
    ax.set_ylim(ylim_min, ylim_max) # Apply the calculated limits
    ## --- END: MODIFICATION 1 ---

    ### 绘制当前价格线
    ax.plot(times_plot, all_pd['close'], label='Close Price') # Added label
    # 填充透明blue, 针对用户一些版本兼容问题进行处理
    # Use times_plot (datetime objects) for fill_between x-axis
    ax.fill_between(times_plot, ylim_min, all_pd['close'], color='blue', alpha=.18, where=all_pd['close']>=ylim_min, interpolate=True)


    ### X轴时间格式化
    # Let matplotlib handle major ticks or customize smarter
    # ax.set_xticks(range(0, len(times_plot), step)) # Using range index is brittle with datetime
    # ax.set_xticklabels(times_plot[::step].strftime(fmt), rotation=45, ha='right') # Apply strftime to datetime index
    fig.autofmt_xdate(rotation=45, ha='right') # Better auto-formatting for dates
    ax.legend() # Display the legend (add labels to plots)
    # plt.tight_layout() # Apply tight_layout later, after annotations

    # Convert order times outside the loop if possible (assuming fmt is constant)
    # This requires orders times to be consistently formatted or datetime objects already
    # Example: order.buy_time = pd.to_datetime(order.buy_time) # If they are strings

    with Progress(len(orders), 0) as pg:
        for index, order in enumerate(orders):
            pg.show(index + 1)

            # Ensure order times are datetime objects for comparison and plotting
            try:
                order_buy_time_dt = pd.to_datetime(str(order.buy_time)) # Use pd.to_datetime for robustness
            except ValueError:
                kd_logger.warning(f"Could not parse order buy_time: {order.buy_time}")
                continue # Skip this order if time is invalid

            # Find the index corresponding to the buy time
            # Use index.get_loc with tolerance for potential floating point/exact match issues if needed
            try:
                # Exact match using the 'trade_time' column (which should be datetime)
                buy_indices = all_pd.index[all_pd['trade_time'] == order_buy_time_dt]
                if buy_indices.empty:
                     # Try finding the closest time if exact match fails
                     buy_loc = all_pd.index.get_indexer([order_buy_time_dt], method='nearest')
                     # Check if the nearest is close enough (e.g., within a minute/hour depending on frequency)
                     time_diff = abs(all_pd.index[buy_loc[0]] - order_buy_time_dt)
                     # Define a reasonable tolerance, e.g., based on kl_pd frequency
                     tolerance = pd.Timedelta(minutes=1) if len(all_pd) > 1 else pd.Timedelta(seconds=0)
                     if len(all_pd)>1:
                         # Estimate frequency if possible
                          freq = pd.infer_freq(all_pd.index)
                          if freq:
                              tolerance = pd.tseries.frequencies.to_offset(freq) / 2 # Half the bar duration
                     
                     if time_diff <= tolerance:
                         st_key = all_pd.iloc[buy_loc[0]]['key']
                         kd_logger.debug(f"Exact buy time not found for {order_buy_time_dt}, using nearest: {all_pd.index[buy_loc[0]]}")
                     else:
                          kd_logger.warning(f"Buy time {order_buy_time_dt} not found in kl_pd index within tolerance {tolerance}.")
                          continue # Skip order
                else:
                     st_key = all_pd.loc[buy_indices[0], 'key'] # Get key using the found index
            except Exception as e:
                 kd_logger.warning(f"Error finding buy_time index for {order_buy_time_dt}: {e}")
                 continue # Skip order

            st_iloc = all_pd.index.get_loc(all_pd[all_pd['key'] == st_key].index[0]) # Get iloc position from key


            if order.sell_type == 'keep':
                # Highlight from buy time to the end
                rv_pd = all_pd.iloc[st_iloc:, :]
            else:
                try:
                    order_sell_time_dt = pd.to_datetime(str(order.sell_time))
                except ValueError:
                    kd_logger.warning(f"Could not parse order sell_time: {order.sell_time}")
                    rv_pd = all_pd.iloc[st_iloc:, :] # Default to 'keep' behavior if sell time invalid?
                    order.sell_type = 'keep' # Mark as keep maybe?
                else:
                     # Find the index corresponding to the sell time
                    try:
                        sell_indices = all_pd.index[all_pd['trade_time'] == order_sell_time_dt]
                        if sell_indices.empty:
                             # Try finding the closest time
                             sell_loc = all_pd.index.get_indexer([order_sell_time_dt], method='nearest')
                             # Check tolerance
                             time_diff = abs(all_pd.index[sell_loc[0]] - order_sell_time_dt)
                             # Define tolerance (reuse from above or define specifically)
                             tolerance = pd.Timedelta(minutes=1) if len(all_pd) > 1 else pd.Timedelta(seconds=0)
                             if len(all_pd)>1:
                                 freq = pd.infer_freq(all_pd.index)
                                 if freq:
                                     tolerance = pd.tseries.frequencies.to_offset(freq) / 2
                             
                             if time_diff <= tolerance:
                                 st_sell_key = all_pd.iloc[sell_loc[0]]['key']
                                 kd_logger.debug(f"Exact sell time not found for {order_sell_time_dt}, using nearest: {all_pd.index[sell_loc[0]]}")

                             else:
                                 kd_logger.warning(f"Sell time {order_sell_time_dt} not found in kl_pd index within tolerance {tolerance}. Ending highlight at last data point.")
                                 rv_pd = all_pd.iloc[st_iloc:, :] # Highlight to end if sell time not found
                                 order.sell_type = 'keep' # Treat as if not sold within data range

                        else:
                             st_sell_key = all_pd.loc[sell_indices[0], 'key']

                        # Only proceed if st_sell_key was successfully found
                        if 'st_sell_key' in locals() and st_sell_key is not None:
                            et_iloc = all_pd.index.get_loc(all_pd[all_pd['key'] == st_sell_key].index[0])
                            rv_pd = all_pd.iloc[st_iloc : et_iloc + 1, :]
                        else:
                             # Handle case where sell time was not found properly
                             rv_pd = all_pd.iloc[st_iloc:, :]
                             order.sell_type = 'keep' # Fallback


                    except Exception as e:
                         kd_logger.warning(f"Error finding sell_time index for {order_sell_time_dt}: {e}")
                         rv_pd = all_pd.iloc[st_iloc:, :] # Default to 'keep' behavior
                         order.sell_type = 'keep'


            # --- Determine win/loss ---
            order_win = False # Default
            try:
                if order.sell_type == 'keep':
                     # Compare buy price to the *last* price in the highlighted range (or overall last price)
                     last_rv_price = rv_pd.iloc[-1].close if not rv_pd.empty else now_price
                     order_win = (last_rv_price - order.buy_price) * order.expect_direction > 0
                elif order.sell_type == 'win':
                    order_win = True
                elif order.sell_type == 'loss': # Assuming 'loss' is the other type
                    order_win = False
                else: # Explicitly check for sold orders if type isn't just 'win'/'loss'
                    if order.sell_price is not None:
                         order_win = (order.sell_price - order.buy_price) * order.expect_direction > 0

                # --- Fill between based on win/loss ---
                fill_color = 'red' if order_win else 'green'
                fill_alpha = 0.18 if order_win else 0.38
                # Use datetime index for fill_between
                plt.fill_between(
                    rv_pd.index,
                    ylim_min, # Use ylim_min as base for fill
                    rv_pd['close'],
                    color=fill_color,
                    alpha=fill_alpha,
                    where=rv_pd['close'] >= ylim_min, # Ensure fill stays within plot
                    interpolate=True) # Interpolate helps with gaps/edges
            except Exception as e: # More specific exception?
                kd_logger.warning(f'fill_between error: {e}')


            # --- Annotation ---
            # Use the actual datetime objects for positioning
            buy_time_dt_for_plot = all_pd.index[st_iloc] # Get the actual datetime from index
            buy_price_at_time = all_pd.loc[buy_time_dt_for_plot, 'close'] # Use close price at that exact time

            # 格式化买入信息标签
            buy_tip = '{:.2f}'.format(order.buy_price)

            ## --- START: MODIFICATION 2 ---
            # Adjust annotation positioning
            # Place text slightly above the data point, relative to the y-range
            y_offset_factor = 0.03 # Percentage of y-range to offset text vertically
            y_buy_text_pos = buy_price_at_time + y_range * y_offset_factor

            # Ensure text position stays within bounds (optional, but safer)
            y_buy_text_pos = max(ylim_min + y_range * 0.01, min(ylim_max - y_range * 0.01, y_buy_text_pos))

            plt.annotate(buy_tip,
                         xy=(buy_time_dt_for_plot, buy_price_at_time), # Arrow points to the data point
                         xytext=(buy_time_dt_for_plot, y_buy_text_pos), # Text positioned slightly above
                         arrowprops=dict(facecolor='red', arrowstyle='->', connectionstyle='arc3,rad=.2'), # Nicer arrow
                         horizontalalignment='center', # Center text horizontally
                         verticalalignment='bottom') # Place text box bottom at xytext y-coord
            ## --- END: MODIFICATION 2 ---


            # --- Sell Annotation ---
            if order.sell_type != 'keep' and order.sell_price is not None:
                sell_time_dt_for_plot = rv_pd.index[-1] # Actual sell time from index
                sell_price_at_time = all_pd.loc[sell_time_dt_for_plot, 'close'] # Actual close price at sell time

                pfr = (order.sell_price - order.buy_price) / order.buy_price if order.buy_price != 0 else 0
                sell_tip = '{:.2f}\n{:.2f}%'.format(order.sell_price, pfr * 100)

                ## --- START: MODIFICATION 3 ---
                # Adjust sell annotation positioning similarly
                y_sell_text_pos = sell_price_at_time + y_range * y_offset_factor
                 # Ensure text position stays within bounds
                y_sell_text_pos = max(ylim_min + y_range * 0.01, min(ylim_max - y_range * 0.01, y_sell_text_pos))

                plt.annotate(sell_tip,
                             xy=(sell_time_dt_for_plot, sell_price_at_time), # Arrow points to data point
                             xytext=(sell_time_dt_for_plot, y_sell_text_pos), # Text positioned slightly above
                             arrowprops=dict(facecolor='green', arrowstyle='->', connectionstyle='arc3,rad=.2'), # Nicer arrow
                             horizontalalignment='center', # Center text
                             verticalalignment='bottom') # Place text bottom at y-coord
                ## --- END: MODIFICATION 3 ---

            elif order.sell_type == 'keep': # Handle 'keep' orders explicitly for annotation
                 # Annotate the current status at the end of the chart
                 last_time_dt = all_pd.index[-1]
                 last_close_price = now_price # Use the already fetched now_price

                 pfr = (now_price - order.buy_price) / order.buy_price if order.buy_price != 0 else 0
                 sell_tip = 'Keep\n{:.2f}\n{:.2f}%'.format(now_price, pfr * 100) # Indicate it's still held

                 ## --- START: MODIFICATION 4 ---
                 # Adjust 'keep' annotation positioning
                 y_sell_text_pos = last_close_price + y_range * y_offset_factor
                 # Ensure text position stays within bounds
                 y_sell_text_pos = max(ylim_min + y_range * 0.01, min(ylim_max - y_range * 0.01, y_sell_text_pos))

                 plt.annotate(sell_tip,
                             xy=(last_time_dt, last_close_price), # Arrow points to last data point
                             xytext=(last_time_dt, y_sell_text_pos), # Text positioned slightly above
                             arrowprops=dict(facecolor='blue', arrowstyle='->', connectionstyle='arc3,rad=.2'), # Different color?
                             horizontalalignment='center', # Center text
                             verticalalignment='bottom') # Place text bottom at y-coord
                 ## --- END: MODIFICATION 4 ---


    # Set title - potentially use the symbol from the *first* order if they are all the same
    if orders:
        plt.title(f"Trade History for {orders[0].buy_symbol}")
    else:
        plt.title("Trade History")

    plt.xlabel("Time") # Add X axis label
    plt.ylabel("Price") # Add Y axis label
    plt.grid(True, linestyle='--', alpha=0.6) # Add grid for readability
    plt.tight_layout() # Apply after all elements are added
    # plt.show() # Add this if you want to display the plot immediately