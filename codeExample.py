// © icefishball

//@version=5
indicator(title='KDJMA', precision=1, overlay=true, max_boxes_count = 10)


// Inputs for optimization
ilong = 9
isig = 3

// KDJ Calculation
bcwsma(s, l, m) =>
    _s = s
    _l = l
    _m = m
    _bcwsma = 0.0
    _bcwsma := (_m * _s + (_l - _m) * nz(_bcwsma[1])) / _l
    _bcwsma

c = close
h = ta.highest(high, ilong)
l = ta.lowest(low, ilong)
RSV = 100 * ((c - l) / (h - l))
pK = bcwsma(RSV, isig, 1)
pD = bcwsma(pK, isig, 1)
pJ = 3 * pK - 2 * pD

//RSI input
ma(source, length, type) =>
    switch type
        "SMA" => ta.sma(source, length)
        "Bollinger Bands" => ta.sma(source, length)
        "EMA" => ta.ema(source, length)
        "SMMA (RMA)" => ta.rma(source, length)
        "WMA" => ta.wma(source, length)
        "VWMA" => ta.vwma(source, length)

rsiLengthInput = 14
rsiSourceInput = close
bbMultInput = 2

up = ta.rma(math.max(ta.change(rsiSourceInput), 0), rsiLengthInput)
down = ta.rma(-math.min(ta.change(rsiSourceInput), 0), rsiLengthInput)
rsi = down == 0 ? 100 : up == 0 ? 0 : 100 - (100 / (1 + up / down))


rs = rsi


// Buy and Sell Signals
reverse = pJ > 80  and pJ < pJ[1] and close > close[1] and rs >60
reverse1 = pJ < 20  and pJ > pJ[1] and close < close[1] and rs <40


plotshape(reverse1, style=shape.labelup, location=location.belowbar, color=color.new(color.green, 0), text='R', textcolor=color.new(color.white, 0))
plotshape(reverse, style=shape.labeldown, location=location.abovebar, color=color.new(color.red, 0), text='R', textcolor=color.new(color.white, 0))

// MA

MA1len = input.int(20,  "MA1 Length", minval=1)
MA2len = input.int(50, "MA2 Length", minval=1)
MA3len = input.int(100,"MA3 Length",minval= 0)
MA1 = ta.sma(close, MA1len)
MA2 = ta.sma(close, MA2len)
MA3 = ta.sma(close, MA3len)
plot(MA1, color = #eeff00, title="MA1")
plot(MA2, color = #ff0000, title="MA2")
plot(MA3, color = #00f000, title="MA3", display=display.none)

/// VWAP ///
vwaplength = input(title='VWAP Length', defval=1)
cvwap = ta.ema(ta.vwap, vwaplength)
plotvwap = plot(cvwap, color=#ff7300, title='VWAP', linewidth=1)



// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © icefishball

//@version=5
indicator("EMAPMO", overlay = true)


//PMO
firstLength = 100
secondLength = 50
signalLength = 10
src = input(title="Source", defval=close)

pmo = ta.ema(10 * ta.ema(nz(ta.roc(src, 1)), firstLength), secondLength)
signal = ta.ema(pmo, signalLength)

//
t = input.string(title = "Time Frame", defval = "1min", options = ["1min", "5min", "15min", "1h", "日"])

updivergence = if t == "1min"
    0.04 
else if t == "5min"
    0.06
else if t == "15min"
    0.16
else if t == "1h"
    0.4
else if t == "日"
    1.5
downdivergence = if t == "1min"
    -0.04
else if t == "5min"
    -0.1
else if t == "15min"
    -0.16
else if t == "1h"
    -0.4
else if t == "日"
    -1
//
backgroundColour = if pmo > -0.003 and pmo < 0.003
    color.new(#ffffff, 90)
else if pmo > 0
    color.new(color.green,90)
else if pmo < 0
    color.new(color.red,90)
bgcolor(backgroundColour)

p = pmo - signal
q = signal - pmo

downdown1 = signal > updivergence and p < p[1] and pmo > signal and p[1] < p[2]
upupup1 = signal < downdivergence and q < q[1] and pmo < signal and q[1] < q[2]

downdown = pmo > updivergence and ta.crossunder(pmo,signal)
upupup = pmo < downdivergence and ta.crossover(pmo,signal)

signaltype = input.string(title = "signaltype", defval = "Normal", options = ["Normal", "Early"] )


plotshape(signaltype == "Normal" and upupup,title = "下方極限",location = location.belowbar,style = shape.labelup, color = color.green, text = "GG", textcolor = color.white)
plotshape(signaltype == "Normal" and downdown,title = "上方極限",location = location.abovebar,style = shape.labeldown, color = color.red, text = "GG", textcolor = color.white)
plotshape(signaltype == "Early" and upupup1,title = "下方極限",location = location.belowbar,style = shape.labelup, color = color.green, text = "G", textcolor = color.white)
plotshape(signaltype == "Early" and downdown1,title = "上方極限",location = location.abovebar,style = shape.labeldown, color = color.red, text = "G", textcolor = color.white)

//ema cloud
matype = 'SMA'

ma_len1 = 99
ma_len2 = 101
ma_len3 = 90
ma_len4 = 110
ma_len5 = 19
ma_len6 = 20
ma_len7 = 17
ma_len8 = 23
ma_len9 = 180
ma_len10 = 200

src2 = input(title='Source', defval=hl2)
ma_offset = input(title='Offset', defval=0)
//res = input(title="Resolution", type=resolution, defval="240")

f_ma(malen) =>
    float result = 0
    if matype == 'EMA'
        result := ta.ema(src2, malen)
        result
    if matype == 'SMA'
        result := ta.sma(src2, malen)
        result
    result

htf_ma1 = f_ma(ma_len1)
htf_ma2 = f_ma(ma_len2)
htf_ma3 = f_ma(ma_len3)
htf_ma4 = f_ma(ma_len4)
htf_ma5 = f_ma(ma_len5)
htf_ma6 = f_ma(ma_len6)
htf_ma7 = f_ma(ma_len7)
htf_ma8 = f_ma(ma_len8)
htf_ma9 = f_ma(ma_len9)
htf_ma10 = f_ma(ma_len10)

//plot(out1, color=green, offset=ma_offset)
//plot(out2, color=red, offset=ma_offset)

//lengthshort = input(8, minval = 1, title = "Short EMA Length")
//lengthlong = input(200, minval = 2, title = "Long EMA Length")
//emacloudleading = input(50, minval = 0, title = "Leading Period For EMA Cloud")
//src = input(hl2, title = "Source")
showlong = input(false, title='Show Long Alerts')
showshort = input(false, title='Show Short Alerts')
showLine = input(false, title='Display EMA Line')
ema1 = input(true, title='Show EMA Cloud-1')
ema2 = input(true, title='Show EMA Cloud-2')
ema3 = input(true, title='Show EMA Cloud-3')
ema4 = input(true, title='Show EMA Cloud-4')
ema5 = input(false, title='Show EMA Cloud-5')

emacloudleading = input.int(0, minval=0, title='Leading Period For EMA Cloud')
mashort1 = htf_ma1
malong1 = htf_ma2
mashort2 = htf_ma3
malong2 = htf_ma4
mashort3 = htf_ma5
malong3 = htf_ma6
mashort4 = htf_ma7
malong4 = htf_ma8
mashort5 = htf_ma9
malong5 = htf_ma10

cloudcolour1 = mashort1 >= malong1 ? #036103 : #880e4f
cloudcolour2 = mashort2 >= malong2 ? #4caf50 : #f44336
cloudcolour3 = mashort3 >= malong3 ? #2196f3 : #ffb74d
cloudcolour4 = mashort4 >= malong4 ? #009688 : #f06292
cloudcolour5 = mashort5 >= malong5 ? #05bed5 : #e65100
//03abc1

mashortcolor1 = mashort1 >= mashort1[1] ? color.olive : color.maroon
mashortcolor2 = mashort2 >= mashort2[1] ? color.olive : color.maroon
mashortcolor3 = mashort3 >= mashort3[1] ? color.olive : color.maroon
mashortcolor4 = mashort4 >= mashort4[1] ? color.olive : color.maroon
mashortcolor5 = mashort5 >= mashort5[1] ? color.rgb(179, 179, 43) : color.maroon


mashortline1 = plot(ema1 ? mashort1 : na, color=showLine ? mashortcolor1 : na, linewidth=1, offset=emacloudleading, title='Short Leading EMA1')
mashortline2 = plot(ema2 ? mashort2 : na, color=showLine ? mashortcolor2 : na, linewidth=1, offset=emacloudleading, title='Short Leading EMA2')
mashortline3 = plot(ema3 ? mashort3 : na, color=showLine ? mashortcolor3 : na, linewidth=1, offset=emacloudleading, title='Short Leading EMA3')
mashortline4 = plot(ema4 ? mashort4 : na, color=showLine ? mashortcolor4 : na, linewidth=1, offset=emacloudleading, title='Short Leading EMA4')
mashortline5 = plot(ema5 ? mashort5 : na, color=showLine ? mashortcolor5 : na, linewidth=1, offset=emacloudleading, title='Short Leading EMA5')

malongcolor1 = malong1 >= malong1[1] ? color.rgb(0, 255, 8) : color.red
malongcolor2 = malong2 >= malong2[1] ? color.rgb(0, 255, 8) : color.red
malongcolor3 = malong3 >= malong3[1] ? color.rgb(0, 255, 8) : color.red
malongcolor4 = malong4 >= malong4[1] ? color.rgb(0, 255, 8) : color.red
malongcolor5 = malong5 >= malong5[1] ? color.rgb(0, 255, 8) : color.red

malongline1 = plot(ema1 ? malong1 : na, color=showLine ? malongcolor1 : na, linewidth=3, offset=emacloudleading, title='Long Leading EMA1')
malongline2 = plot(ema2 ? malong2 : na, color=showLine ? malongcolor2 : na, linewidth=3, offset=emacloudleading, title='Long Leading EMA2')
malongline3 = plot(ema3 ? malong3 : na, color=showLine ? malongcolor3 : na, linewidth=3, offset=emacloudleading, title='Long Leading EMA3')
malongline4 = plot(ema4 ? malong4 : na, color=showLine ? malongcolor4 : na, linewidth=3, offset=emacloudleading, title='Long Leading EMA4')
malongline5 = plot(ema5 ? malong5 : na, color=showLine ? malongcolor5 : na, linewidth=3, offset=emacloudleading, title='Long Leading EMA5')


// Updated fill function calls using color.new() for transparency
fill(mashortline1, malongline1, color=color.new(cloudcolour1, 45), title='MA Cloud1')
fill(mashortline2, malongline2, color=color.new(cloudcolour2, 65), title='MA Cloud2')
fill(mashortline3, malongline3, color=color.new(cloudcolour3, 70), title='MA Cloud3')
fill(mashortline4, malongline4, color=color.new(cloudcolour4, 65), title='MA Cloud4')




