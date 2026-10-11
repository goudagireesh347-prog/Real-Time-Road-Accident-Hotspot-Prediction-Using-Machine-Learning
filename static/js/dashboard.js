/**
 * RoadSafe AI – Real-Time Accident Hotspot Prediction System
 * JavaScript Controller for dynamic charts, API integration, and UI interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    // Check if on Analytics page
    if (document.getElementById('analyticsChartsWrapper')) {
        loadAnalyticsCharts();
    }

    // Check if on Dashboard page
    if (document.getElementById('dashboardTrendChart')) {
        loadDashboardSummaryCharts();
    }
});

/**
 * Loads and renders the 8 interactive charts on the Analytics page
 */
function loadAnalyticsCharts() {
    fetch('/api/analytics-data')
        .then(response => response.json())
        .then(res => {
            if (res.status !== 'success') {
                console.error('Failed to load analytics data:', res);
                return;
            }
            const data = res;

            // 1. Accidents by Hour
            const hourCtx = document.getElementById('chartHour')?.getContext('2d');
            if (hourCtx) {
                const hours = Object.keys(data.by_hour).map(h => `${h}:00`);
                const hourValues = Object.values(data.by_hour);
                new Chart(hourCtx, {
                    type: 'line',
                    data: {
                        labels: hours,
                        datasets: [{
                            label: 'Accidents by Hour of Day',
                            data: hourValues,
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.15)',
                            fill: true,
                            tension: 0.35,
                            pointBackgroundColor: '#1d4ed8'
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }

            // 2. Accidents by Day of Week
            const dayCtx = document.getElementById('chartDay')?.getContext('2d');
            if (dayCtx) {
                new Chart(dayCtx, {
                    type: 'bar',
                    data: {
                        labels: Object.keys(data.by_day),
                        datasets: [{
                            label: 'Accidents by Day',
                            data: Object.values(data.by_day),
                            backgroundColor: '#6366f1',
                            borderRadius: 6
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }

            // 3. Accidents by Month
            const monthCtx = document.getElementById('chartMonth')?.getContext('2d');
            if (monthCtx) {
                new Chart(monthCtx, {
                    type: 'bar',
                    data: {
                        labels: Object.keys(data.by_month),
                        datasets: [{
                            label: 'Monthly Accident Count',
                            data: Object.values(data.by_month),
                            backgroundColor: '#0ea5e9',
                            borderRadius: 6
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }

            // 4. Accidents by Weather
            const weatherCtx = document.getElementById('chartWeather')?.getContext('2d');
            if (weatherCtx) {
                new Chart(weatherCtx, {
                    type: 'doughnut',
                    data: {
                        labels: Object.keys(data.by_weather),
                        datasets: [{
                            data: Object.values(data.by_weather),
                            backgroundColor: ['#38bdf8', '#3b82f6', '#64748b', '#0284c7', '#1e293b']
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }

            // 5. Accidents by Road Condition
            const roadCtx = document.getElementById('chartRoad')?.getContext('2d');
            if (roadCtx) {
                new Chart(roadCtx, {
                    type: 'bar',
                    data: {
                        labels: Object.keys(data.by_road_condition),
                        datasets: [{
                            label: 'Road Condition',
                            data: Object.values(data.by_road_condition),
                            backgroundColor: '#f59e0b',
                            borderRadius: 6
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }

            // 6. Accidents by Severity
            const sevCtx = document.getElementById('chartSeverity')?.getContext('2d');
            if (sevCtx) {
                new Chart(sevCtx, {
                    type: 'pie',
                    data: {
                        labels: Object.keys(data.by_severity),
                        datasets: [{
                            data: Object.values(data.by_severity),
                            backgroundColor: ['#ef4444', '#f97316', '#eab308', '#10b981']
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }

            // 7. Accidents by Location (Top 10)
            const locCtx = document.getElementById('chartLocation')?.getContext('2d');
            if (locCtx) {
                new Chart(locCtx, {
                    type: 'bar',
                    data: {
                        labels: Object.keys(data.by_location),
                        datasets: [{
                            label: 'Accidents',
                            data: Object.values(data.by_location),
                            backgroundColor: '#8b5cf6',
                            borderRadius: 6
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
            }

            // 8. High-Risk Locations
            const highCtx = document.getElementById('chartHighRisk')?.getContext('2d');
            if (highCtx) {
                new Chart(highCtx, {
                    type: 'bar',
                    data: {
                        labels: Object.keys(data.high_risk_locations),
                        datasets: [{
                            label: 'High-Risk Incident Count',
                            data: Object.values(data.high_risk_locations),
                            backgroundColor: '#dc2626',
                            borderRadius: 6
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
            }
        })
        .catch(err => console.error('Error fetching analytics data:', err));
}

/**
 * Loads preview trend chart on Dashboard
 */
function loadDashboardSummaryCharts() {
    fetch('/api/analytics-data')
        .then(res => res.json())
        .then(data => {
            const ctx = document.getElementById('dashboardTrendChart')?.getContext('2d');
            if (!ctx || data.status !== 'success') return;

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Object.keys(data.by_hour).map(h => `${h}:00`),
                    datasets: [{
                        label: 'Hourly Incident Rate',
                        data: Object.values(data.by_hour),
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        fill: true,
                        tension: 0.35
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } }
                }
            });
        })
        .catch(err => console.error('Error loading dashboard chart:', err));
}

/**
 * Pre-fills the Real-Time Predict form with selected Zone coordinates & typical traffic
 */
function prefillZone(name, lat, lon, speed, roadType, traffic) {
    const latField = document.getElementById('latitude');
    const lonField = document.getElementById('longitude');
    const speedField = document.getElementById('speed_limit');
    const roadTypeField = document.getElementById('road_type');
    const trafficField = document.getElementById('traffic_volume');

    if (latField) latField.value = lat;
    if (lonField) lonField.value = lon;
    if (speedField) speedField.value = speed;
    if (roadTypeField) roadTypeField.value = roadType;
    if (trafficField && traffic) trafficField.value = traffic;
}
