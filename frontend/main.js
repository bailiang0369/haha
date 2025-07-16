$(document).ready(function() {
    const chartOptions = {
        width: 800,
        height: 500,
        layout: {
            backgroundColor: '#ffffff',
            textColor: 'rgba(33, 56, 77, 1)',
        },
        grid: {
            vertLines: {
                color: 'rgba(197, 203, 206, 0.5)',
            },
            horzLines: {
                color: 'rgba(197, 203, 206, 0.5)',
            },
        },
        crosshair: {
            mode: window.LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: 'rgba(197, 203, 206, 0.8)',
        },
        timeScale: {
            borderColor: 'rgba(197, 203, 206, 0.8)',
        },
    };

    const chart = window.LightweightCharts.createChart(document.getElementById('chart-container'), chartOptions);

    const candleSeries = chart.addCandlestickSeries({
        upColor: 'rgba(255, 144, 0, 1)',
        downColor: '#000',
        borderDownColor: 'rgba(255, 144, 0, 1)',
        borderUpColor: 'rgba(255, 144, 0, 1)',
        wickDownColor: 'rgba(255, 144, 0, 1)',
        wickUpColor: 'rgba(255, 144, 0, 1)',
    });

    // Fetch and display K-lines
    fetch('http://localhost:8000/api/klines')
        .then(response => response.json())
        .then(data => {
            candleSeries.setData(data);
        })
        .catch(error => console.error('Error fetching klines:', error));

    // Fetch and display MACD
    fetch('http://localhost:8000/api/macd')
        .then(response => response.json())
        .then(data => {
            const macdLine = chart.addLineSeries({
                color: '#2962FF',
                lineWidth: 2,
                priceScaleId: 'macd',
            });
            const signalLine = chart.addLineSeries({
                color: '#FF6D00',
                lineWidth: 2,
                priceScaleId: 'macd',
            });
            const histogramSeries = chart.addHistogramSeries({
                color: '#26a69a',
                priceFormat: {
                    type: 'volume',
                },
                priceScaleId: 'macd',
            });

            const macdData = data.map(d => ({ time: d.time, value: d.macd }));
            const signalData = data.map(d => ({ time: d.time, value: d.signal }));
            const histogramData = data.map(d => ({
                time: d.time,
                value: d.histogram,
                color: d.histogram >= 0 ? 'rgba(0, 150, 136, 0.8)' : 'rgba(255, 82, 82, 0.8)',
            }));

            chart.priceScale('macd').applyOptions({
                scaleMargins: {
                    top: 0.8,
                    bottom: 0,
                },
            });

            macdLine.setData(macdData);
            signalLine.setData(signalData);
            histogramSeries.setData(histogramData);
        })
        .catch(error => console.error('Error fetching MACD:', error));
});
