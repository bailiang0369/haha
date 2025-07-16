$(document).ready(function() {
    const chart = LightweightCharts.createChart(document.getElementById('chart-container'), {
        width: 800,
        height: 600,
        // ... other chart options ...
    });

    const candleSeries = chart.addCandlestickSeries({
        pane: 0,
    });

    const histogramSeries = chart.addHistogramSeries({
        color: '#26a69a',
        pane: 1,
    });

    // Fetch and display K-lines
    fetch('http://localhost:8000/api/klines')
        .then(res => res.json())
        .then(data => {
            candleSeries.setData(data);
        })
        .catch(error => console.error('Error fetching klines:', error));

    // Fetch and display MACD Histogram
    fetch('http://localhost:8000/api/macd')
        .then(res => res.json())
        .then(data => {
            const histogramData = data.map(d => ({
                time: d.time,
                value: d.histogram,
                color: d.histogram >= 0 ? 'rgba(0, 150, 136, 0.8)' : 'rgba(255, 82, 82, 0.8)',
            }));
            histogramSeries.setData(histogramData);
        })
        .catch(error => console.error('Error fetching MACD:', error));
});
