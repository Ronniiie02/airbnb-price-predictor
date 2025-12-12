// Global variables
let currentTab = 'analysis';
let filterOptions = {};
let models = {};
let selectedModel = 'Random Forest';
let dataSummary = {};
let currentFilters = null; // ✅ 关键：保存当前筛选条件

// Color palette
const colors = {
    primary: '#FF5A5F',
    secondary: '#00A699',
    accent: '#FC642D',
    warning: '#FFB400',
    blue: '#1f77b4',
    green: '#2ca02c',
    purple: '#9467bd'
};

// --------------------------------------------------
// Initialization
// --------------------------------------------------

document.addEventListener('DOMContentLoaded', function () {
    initializeApp();
});

async function initializeApp() {
    try {
        await Promise.all([
            loadDataSummary(),
            loadFilterOptions(),
            loadModels()
        ]);

        if (dataSummary.best_model && dataSummary.best_model.name) {
            selectedModel = dataSummary.best_model.name;
        }

        updateSummaryCards();
        populateFilters();
        populatePredictionForm();
        createModelCards();

        // 初次加载：无 filters
        currentFilters = null;
        await loadAllCharts(currentFilters);

        console.log('Application initialized successfully');
    } catch (error) {
        console.error('Error initializing app:', error);
        showError('Failed to initialize application. Please refresh the page.');
    }
}

// --------------------------------------------------
// Data loading
// --------------------------------------------------

async function loadDataSummary() {
    const response = await fetch('/api/data/summary');
    if (!response.ok) throw new Error('Failed to load data summary');
    dataSummary = await response.json();
}

async function loadFilterOptions() {
    const response = await fetch('/api/filters');
    if (!response.ok) throw new Error('Failed to load filter options');
    filterOptions = await response.json();
}

async function loadModels() {
    const response = await fetch('/api/models');
    if (!response.ok) throw new Error('Failed to load models');
    models = await response.json();
}

// --------------------------------------------------
// UI: Executive summary
// --------------------------------------------------

function updateSummaryCards() {
    if (!dataSummary || !dataSummary.total_listings) return;

    document.getElementById('total-listings').textContent =
        Number(dataSummary.total_listings).toLocaleString();
    document.getElementById('avg-price').textContent =
        `$${Math.round(Number(dataSummary.avg_price || 0))}`;

    const pr = dataSummary.price_range || dataSummary.priceRange || {};
    if (pr.min != null && pr.max != null) {
        document.getElementById('summary-price-range').textContent =
            `$${Math.round(pr.min)}-$${Math.round(pr.max)}`;
    }

    const mlCount =
        dataSummary.ml_models_trained ??
        dataSummary.ml_models ??
        dataSummary.mlModelsTrained ??
        dataSummary.models_trained ??
        null;

    document.getElementById('summary-models').textContent =
        (mlCount != null ? String(mlCount) : '-');

    const best = dataSummary.best_model || dataSummary.bestModel || null;
    const avgMae = dataSummary.avg_mae ?? dataSummary.avgMAE ?? null;
    const mae = (best && best.mae != null) ? best.mae : avgMae;

    document.getElementById('summary-mae').textContent =
        (mae != null ? `±$${Number(mae).toFixed(0)}` : '±$--');

    const insightEl = document.getElementById('key-insight-text');
    if (mae != null) {
        const modelName = (best && best.name) ? best.name : 'best-performing model';
        const pctOfAvg = dataSummary.avg_price ? Math.round(100 * mae / dataSummary.avg_price) : null;

        insightEl.innerHTML = `
            <strong>Key Insight:</strong>
            Using <span class="font-semibold">${modelName}</span>, the average error is about
            <span class="font-semibold">$${Number(mae).toFixed(0)}</span>${pctOfAvg != null ? ` (~${pctOfAvg}% of avg price)` : ''}.
            Location and room type drive most of the variation; review activity and availability are secondary signals.
        `;
    } else {
        insightEl.innerHTML = `<strong>Key Insight:</strong> Loading key insight based on model performance...`;
    }
}

// --------------------------------------------------
// UI: Filters & prediction form
// --------------------------------------------------

function populateFilters() {
    const groupSelect = document.getElementById('filter-group');
    groupSelect.innerHTML = filterOptions.neighbourhood_groups
        .map(g => `<option value="${g}">${g}</option>`)
        .join('');

    const roomSelect = document.getElementById('filter-room');
    roomSelect.innerHTML = filterOptions.room_types
        .map(r => `<option value="${r}">${r}</option>`)
        .join('');

    document.getElementById('price-min').value = Math.round(filterOptions.price_range.min);
    document.getElementById('price-max').value = Math.round(Math.min(filterOptions.price_range.max, 500));

    document.getElementById('nights-min').value = filterOptions.minimum_nights_range.min;
    document.getElementById('nights-max').value = Math.min(30, filterOptions.minimum_nights_range.max);

    document.getElementById('filtered-count').textContent = dataSummary.total_listings.toLocaleString();
    document.getElementById('total-count').textContent = dataSummary.total_listings.toLocaleString();
}

function populatePredictionForm() {
    const predGroupSelect = document.getElementById('pred-group');
    predGroupSelect.innerHTML = filterOptions.neighbourhood_groups
        .map(g => `<option value="${g}">${g}</option>`)
        .join('');

    const predRoomSelect = document.getElementById('pred-room');
    predRoomSelect.innerHTML = filterOptions.room_types
        .map(r => `<option value="${r}">${r}</option>`)
        .join('');

    updateNeighbourhoods();
}

// --------------------------------------------------
// UI: Model cards
// --------------------------------------------------

function createModelCards() {
    const container = document.getElementById('model-cards');
    container.innerHTML = Object.entries(models).map(([name, model]) => `
        <div class="card card-hover rounded-lg p-4 cursor-pointer ${name === selectedModel ? 'ring-2 ring-blue-500' : ''}"
             onclick="selectModel('${name}')">
            <h5 class="font-bold text-gray-800 mb-2">${name}</h5>
            <div class="text-sm text-gray-600">
                <div>MAE: $${model.mae.toFixed(2)}</div>
                <div>RMSE: $${model.rmse.toFixed(2)}</div>
            </div>
        </div>
    `).join('');
}

// --------------------------------------------------
// Charts
// --------------------------------------------------

async function loadAllCharts(filters = null) {
    try {
        await Promise.all([
            loadPriceDistributionChart(filters),
            loadPriceByRoomTypeChart(filters),
            loadPriceByNeighbourhoodChart(filters),
            loadReviewsVsPriceChart(filters),
            loadListingsMapChart(filters),
            loadPriceHeatmapChart(filters),
            loadModelPerformanceChart(filters),
        ]);
    } catch (err) {
        console.error('Error loading charts:', err);
        showError('Failed to load some charts. Please try again.');
    }
}

async function postChart(endpoint, filters) {
    const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filters ?? {})
    });
    if (!response.ok) throw new Error(`Failed chart: ${endpoint}`);
    return await response.json(); // returns fig.to_json() string
}

async function loadPriceDistributionChart(filters = null) {
    try {
        const chartData = await postChart('/api/charts/price-distribution', filters);
        const plotData = JSON.parse(chartData);
        if (plotData.data[0]) plotData.data[0].marker = { color: colors.primary };
        Plotly.newPlot('price-distribution-chart', plotData.data, plotData.layout, {
            responsive: true, displayModeBar: false
        });
    } catch (err) {
        console.error('Error loading price distribution chart', err);
    }
}

async function loadPriceByRoomTypeChart(filters = null) {
    try {
        const chartData = await postChart('/api/charts/price-by-room-type', filters);
        const plotData = JSON.parse(chartData);
        plotData.data.forEach(trace => { trace.marker = { color: colors.secondary }; });
        Plotly.newPlot('price-room-type-chart', plotData.data, plotData.layout, {
            responsive: true, displayModeBar: false
        });
    } catch (err) {
        console.error('Error loading price by room type chart', err);
    }
}

async function loadPriceByNeighbourhoodChart(filters = null) {
    try {
        const chartData = await postChart('/api/charts/price-by-neighbourhood', filters);
        const plotData = JSON.parse(chartData);
        plotData.data.forEach(trace => { trace.line = { color: colors.accent }; });
        Plotly.newPlot('price-neighbourhood-chart', plotData.data, plotData.layout, {
            responsive: true, displayModeBar: false
        });
    } catch (err) {
        console.error('Error loading price by neighbourhood chart', err);
    }
}

async function loadReviewsVsPriceChart(filters = null) {
    try {
        const chartData = await postChart('/api/charts/reviews-vs-price', filters);
        const plotData = JSON.parse(chartData);
        if (plotData.data[0]) plotData.data[0].marker = { color: colors.warning, opacity: 0.6 };
        Plotly.newPlot('reviews-price-chart', plotData.data, plotData.layout, {
            responsive: true, displayModeBar: false
        });
    } catch (err) {
        console.error('Error loading reviews vs price chart', err);
    }
}

async function loadListingsMapChart(filters = null) {
    try {
        const chartData = await postChart('/api/charts/listings-map', filters);
        const plotData = JSON.parse(chartData);
        Plotly.newPlot('listings-map-chart', plotData.data, plotData.layout, {
            responsive: true, displayModeBar: false
        });
    } catch (err) {
        console.error('Error loading listings map chart', err);
    }
}

async function loadPriceHeatmapChart(filters = null) {
    try {
        const chartData = await postChart('/api/charts/price-heatmap', filters);
        const plotData = JSON.parse(chartData);
        Plotly.newPlot('price-heatmap-chart', plotData.data, plotData.layout, {
            responsive: true, displayModeBar: false
        });
    } catch (err) {
        console.error('Error loading price heatmap chart', err);
    }
}

async function loadModelPerformanceChart(filters = null) {
    try {
        const response = await fetch('/api/model-performance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: selectedModel,
                filters: filters ?? null,
                max_points: 600
            })
        });
        if (!response.ok) throw new Error('Failed to load model performance');
        const chartData = await response.json();
        const plotData = JSON.parse(chartData);

        if (plotData.data[0]) {
            plotData.data[0].marker = { color: colors.primary, size: 6, opacity: 0.6 };
        }

        Plotly.newPlot('model-performance-chart', plotData.data, plotData.layout, {
            responsive: true,
            displayModeBar: false
        });
    } catch (err) {
        console.error('Error loading model performance chart', err);
    }
}

// --------------------------------------------------
// Tabs
// --------------------------------------------------

function switchTab(tabName, buttonEl) {
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    buttonEl.classList.add('active');

    document.querySelectorAll('.tab-content').forEach(content => content.classList.add('hidden'));
    document.getElementById(`${tabName}-tab`).classList.remove('hidden');

    currentTab = tabName;

    setTimeout(() => {
        Plotly.Plots.resize('price-distribution-chart');
        Plotly.Plots.resize('model-performance-chart');
    }, 100);
}

// --------------------------------------------------
// Filters
// --------------------------------------------------

async function applyFilters() {
    try {
        const filters = {
            neighbourhood_groups: Array.from(document.getElementById('filter-group').selectedOptions).map(o => o.value),
            room_types: Array.from(document.getElementById('filter-room').selectedOptions).map(o => o.value)
        };

        const priceMinStr = document.getElementById('price-min').value;
        const priceMaxStr = document.getElementById('price-max').value;
        if (priceMinStr !== '' && priceMaxStr !== '') {
            filters.price_range = [parseFloat(priceMinStr), parseFloat(priceMaxStr)];
        }

        const nightsMinStr = document.getElementById('nights-min').value;
        const nightsMaxStr = document.getElementById('nights-max').value;
        if (nightsMinStr !== '' && nightsMaxStr !== '') {
            filters.minimum_nights_range = [parseInt(nightsMinStr, 10), parseInt(nightsMaxStr, 10)];
        }

        const response = await fetch('/api/data/filter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(filters)
        });
        if (!response.ok) throw new Error('Failed to apply filters');

        const result = await response.json();

        document.getElementById('filtered-count').textContent = result.filtered_count.toLocaleString();
        document.getElementById('total-count').textContent = result.total_count.toLocaleString();

        // ✅ 保存当前 filters，并用它刷新图表
        currentFilters = filters;
        await loadAllCharts(currentFilters);

        showSuccess('Filters applied successfully!');
    } catch (err) {
        console.error('Error applying filters:', err);
        showError('Failed to apply filters. Please try again.');
    }
}

// --------------------------------------------------
// Prediction
// --------------------------------------------------

async function updateNeighbourhoods() {
    const selectedGroup = document.getElementById('pred-group').value;
    const neighbourhoodSelect = document.getElementById('pred-neighbourhood');

    try {
        const response = await fetch(`/api/neighbourhoods/${encodeURIComponent(selectedGroup)}`);
        if (!response.ok) throw new Error('Failed to load neighbourhoods');

        const neighbourhoods = await response.json();
        neighbourhoodSelect.innerHTML = neighbourhoods.map(n => `<option value="${n}">${n}</option>`).join('');
    } catch (err) {
        console.error('Error updating neighbourhoods:', err);
    }
}

function selectModel(modelName) {
    selectedModel = modelName;
    createModelCards();
    // ✅ 切换模型立即刷新 performance 图
    loadModelPerformanceChart(currentFilters);
}

async function predictPrice() {
    try {
        const resultsDiv = document.getElementById('prediction-results');
        resultsDiv.innerHTML = `
            <div class="text-center">
                <div class="loading mx-auto mb-4"></div>
                <p>Predicting price...</p>
            </div>
        `;

        const features = {
            neighbourhood_group: document.getElementById('pred-group').value,
            neighbourhood: document.getElementById('pred-neighbourhood').value,
            room_type: document.getElementById('pred-room').value,
            minimum_nights: parseInt(document.getElementById('pred-min-nights').value, 10),
            number_of_reviews: parseInt(document.getElementById('pred-reviews').value, 10),
            reviews_per_month: parseFloat(document.getElementById('pred-rpm').value),
            calculated_host_listings_count: parseInt(document.getElementById('pred-host-count').value, 10),
            availability_365: parseInt(document.getElementById('pred-availability').value, 10)
        };

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: selectedModel, features })
        });
        if (!response.ok) throw new Error('Failed to predict price');

        const result = await response.json();

        resultsDiv.innerHTML = `
            <div class="text-center">
                <div class="text-4xl font-bold text-blue-600 mb-2">
                    $${result.predicted_price.toFixed(2)}
                </div>
                <div class="text-gray-600 mb-4">Predicted price per night</div>

                <div class="bg-blue-50 rounded-lg p-4 mb-4">
                    <div class="text-sm text-gray-600 mb-2">Recommended range:</div>
                    <div class="font-semibold">
                        $${result.price_range.min.toFixed(0)} - $${result.price_range.max.toFixed(0)}
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-4 text-sm">
                    <div class="bg-gray-50 rounded-lg p-3">
                        <div class="text-gray-600">Percentile</div>
                        <div class="font-semibold">${result.percentile.toFixed(1)}th</div>
                    </div>
                    <div class="bg-gray-50 rounded-lg p-3">
                        <div class="text-gray-600">Model</div>
                        <div class="font-semibold">${result.model}</div>
                    </div>
                </div>

                <div class="mt-4 p-3 bg-yellow-50 rounded-lg text-xs text-gray-600">
                    <i class="fas fa-info-circle mr-1"></i>
                    This price is higher than ${result.percentile.toFixed(1)}% of similar listings
                </div>
            </div>
        `;

        // ✅ 预测后刷新 performance 图（现在是真实数据，不是 demo）
        await loadModelPerformanceChart(currentFilters);

        showSuccess('Price predicted successfully!');
    } catch (err) {
        console.error('Error predicting price:', err);
        showError('Failed to predict price. Please try again.');

        document.getElementById('prediction-results').innerHTML = `
            <div class="text-center text-gray-500">
                <i class="fas fa-exclamation-triangle text-4xl text-red-300 mb-4"></i>
                <p>Failed to predict price. Please check your inputs and try again.</p>
            </div>
        `;
    }
}

// --------------------------------------------------
// Notifications & resize
// --------------------------------------------------

function showSuccess(message) {
    const notification = document.createElement('div');
    notification.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50 fade-in';
    notification.innerHTML = `
        <div class="flex items-center">
            <i class="fas fa-check-circle mr-2"></i>
            <span>${message}</span>
        </div>
    `;
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 3000);
}

function showError(message) {
    const notification = document.createElement('div');
    notification.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg z-50 fade-in';
    notification.innerHTML = `
        <div class="flex items-center">
            <i class="fas fa-exclamation-circle mr-2"></i>
            <span>${message}</span>
        </div>
    `;
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 5000);
}

window.addEventListener('resize', function () {
    setTimeout(() => {
        const chartIds = [
            'price-distribution-chart',
            'price-room-type-chart',
            'price-neighbourhood-chart',
            'reviews-price-chart',
            'listings-map-chart',
            'price-heatmap-chart',
            'model-performance-chart'
        ];
        chartIds.forEach(id => {
            const el = document.getElementById(id);
            if (el) Plotly.Plots.resize(id);
        });
    }, 100);
});
