$(document).ready(function() {
    const chart = window.LightweightCharts.createChart(document.getElementById('chart-container'), {
        width: 800,
        height: 600,
    });

    // Main pane for K-lines
    const mainSeries = chart.addCandlestickSeries();

    // Create a new pane for MACD
    const macdPane = chart.createPane({
        height: 150,
        title: 'MACD',
    });

    const macdLine = chart.addLineSeries({ pane: macdPane, color: '#2962FF', lineWidth: 2 });
    const signalLine = chart.addLineSeries({ pane: macdPane, color: '#FF6D00', lineWidth: 2 });
    const histogramSeries = chart.addHistogramSeries({ pane: macdPane, color: '#26a69a' });

    // Create a new pane for KDJ
    const kdjPane = chart.createPane({
        height: 150,
        title: 'KDJ',
    });

    const kLine = chart.addLineSeries({ pane: kdjPane, color: '#FFC107', lineWidth: 2 });
    const dLine = chart.addLineSeries({ pane: kdjPane, color: '#2196F3', lineWidth: 2 });
    const jLine = chart.addLineSeries({ pane: kdjPane, color: '#F44336', lineWidth: 2 });


    // Fetch and display K-lines
    fetch('http://localhost:8000/api/klines')
        .then(res => res.json())
        .then(data => mainSeries.setData(data));

    // Fetch and display MACD
    fetch('http://localhost:8000/api/macd')
        .then(res => res.json())
        .then(data => {
            macdLine.setData(data.map(d => ({ time: d.time, value: d.macd })));
            signalLine.setData(data.map(d => ({ time: d.time, value: d.signal })));
            histogramSeries.setData(data.map(d => ({
                time: d.time,
                value: d.histogram,
                color: d.histogram >= 0 ? 'rgba(0, 150, 136, 0.8)' : 'rgba(255, 82, 82, 0.8)',
            })));
        });

    // Fetch and display KDJ
    fetch('http://localhost:8000/api/kdj')
        .then(res => res.json())
        .then(data => {
            kLine.setData(data.map(d => ({ time: d.time, value: d.k })));
            dLine.setData(data.map(d => ({ time: d.time, value: d.d })));
            jLine.setData(data.map(d => ({ time: d.time, value: d.j })));
        });
});
