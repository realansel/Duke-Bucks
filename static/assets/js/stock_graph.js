async function fetchStockData(symbol) {
    const response = await fetch(`/get_stock_data/${symbol}`);
    const stockData = await response.json();
    return stockData;
  }
  
async function plotStockData(symbol) {
const stockData = await fetchStockData(symbol);
const trace = {
    x: stockData.map((row) => row.date),
    y: stockData.map((row) => row.close),
    mode: "lines",
    name: symbol,
};

const data = [trace];
const layout = {
    title: `${symbol} Data`,
    xaxis: { title: "Date" },
    yaxis: { title: "Adjusted Close" },
};

Plotly.newPlot("stock-data-plot", data, layout);
}
  
  // Get the symbol from the URL and call the plotStockData function
  const urlParams = new URLSearchParams(window.location.search);
  const symbol = urlParams.get("symbol");
  plotStockData(symbol);
  