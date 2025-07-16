$(document).ready(function() {
    const chartOptions = {
        width: 800,
        height: 600, // Increased height to accommodate panes
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

    // Fetch and display MACD in a new pane
    fetch('http://localhost:8000/api/macd')
        .then(response => response.json())
        .then(data => {
            const macdLine = chart.addLineSeries({
                color: '#2962FF',
                lineWidth: 2,
                pane: 1, // Pane 1
            });
            const signalLine = chart.addLineSeries({
                color: '#FF6D00',
                lineWidth: 2,
                pane: 1, // Pane 1
            });
            const histogramSeries = chart.addHistogramSeries({
                color: '#26a69a',
                priceFormat: {
                    type: 'volume',
                },
                pane: 1, // Pane 1
            });

            const macdData = data.map(d => ({ time: d.time, value: d.macd }));
            const signalData = data.map(d => ({ time: d.time, value: d.signal }));
            const histogramData = data.map(d => ({
                time: d.time,
                value: d.histogram,
                color: d.histogram >= 0 ? 'rgba(0, 150, 136, 0.8)' : 'rgba(255, 82, 82, 0.8)',
            }));

            macdLine.setData(macdData);
            signalLine.setData(signalData);
            histogramSeries.setData(histogramData);
        })
        .catch(error => console.error('Error fetching MACD:', error));

    // Fetch and display KDJ in another new pane
    fetch('http://localhost:8000/api/kdj')
        .then(response => response.json())
        .then(data => {
            const kLine = chart.addLineSeries({
                color: '#FFC107',
                lineWidth: 2,
                pane: 2, // Pane 2
            });
            const dLine = chart.addLineSeries({
                color: '#2196F3',
                lineWidth: 2,
                pane: 2, // Pane 2
            });
            const jLine = chart.addLineSeries({
                color: '#F44336',
                lineWidth: 2,
                pane: 2, // Pane 2
            });

            const kData = data.map(d => ({ time: d.time, value: d.k }));
            const dData = data.map(d => ({ time: d.time, value: d.d }));
            const jData = data.map(d => ({ time: d.time, value: d.j }));

            kLine.setData(kData);
            dLine.setData(dData);
            jLine.setData(jData);
        })
        .catch(error => console.error('Error fetching KDJ:', error));
});
