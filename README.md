# Start State

## Build the base
- [x] Identify scrips
- [x] Get live scrip prices
- [x] Get lot sizes (for NSE)
- [x] Get option chains
- [ ] Gather historical data
- [ ] Set `strategy` based on `conditions` per scrip


## Set the trades
- [ ] Get current positions
- [ ] Get open orders
- [ ] Get closing trades for current positions without open orders
- [ ] Place closing trades
- [ ] Get scrip targets based on `strategy` and `conditions`
- [ ] Get best option trades
- [ ] Place new option orders

# Event Driven

## Upon Order Exection   
- [ ] Journal the scrip execution with the `strategy` and its `condition`
- [ ] Adjust prices of open order that are not closing trades
- [ ] Set closing trades

## Upon Margin Breach
- [ ] Cancel all open orders without underlyings
- [ ] Stop all new order creation
- [ ] Calculate quantum of margin breach
- [ ] Identify top margin breachers
- [ ] Liquidate margin breachers
- [ ] Cancel open orders of liquidated margin breachers
- [ ] Journal margin breachers

## Upon IB Technical Error
- [ ] Journal and inform

## Upon Program Error
- [ ] Journal and inform