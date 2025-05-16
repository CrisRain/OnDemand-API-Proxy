// 全局变量
let refreshInterval = 60; // 默认刷新间隔（秒）
let autoRefreshEnabled = true; // 默认启用自动刷新
let chartInstances = {}; // 存储图表实例的对象
let darkModeEnabled = localStorage.getItem('theme') === 'dark'; // 深色模式状态

// 格式化大数值的函数
function formatChartNumber(value) {
    if (value >= 1000000000) {
        return (value / 1000000000).toFixed(1) + 'G';
    } else if (value >= 1000000) {
        return (value / 1000000).toFixed(1) + 'M';
    } else if (value >= 1000) {
        return (value / 1000).toFixed(1) + 'K';
    }
    return value;
}

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化图表
    initializeCharts();
    
    // 设置自动刷新
    setupAutoRefresh();
    
    // 主题切换
    setupThemeToggle();
    
    // 加载保存的主题
    loadSavedTheme();
    
    // 添加表格交互功能
    enhanceTableInteraction();
    
    // 添加保存统计数据按钮事件
    setupSaveStatsButton();
    
    // 更新页脚信息
    updateFooterInfo();
    
    // 表格排序和筛选
    const table = document.getElementById('history-table');
    if (table) {
        const headers = table.querySelectorAll('th[data-sort]');
        const rows = Array.from(table.querySelectorAll('tbody tr'));
        const rowsPerPage = 10;
        let currentPage = 1;
        let filteredRows = [...rows];
        
        // 初始化分页
        function initPagination() {
            const totalPages = Math.ceil(filteredRows.length / rowsPerPage);
            document.getElementById('total-pages').textContent = totalPages;
            document.getElementById('current-page').textContent = currentPage;
            document.getElementById('prev-page').disabled = currentPage === 1;
            document.getElementById('next-page').disabled = currentPage === totalPages || totalPages === 0;
            
            // 显示当前页的行
            const startIndex = (currentPage - 1) * rowsPerPage;
            const endIndex = startIndex + rowsPerPage;
            
            rows.forEach(row => row.style.display = 'none');
            filteredRows.slice(startIndex, endIndex).forEach(row => row.style.display = '');
        }
        
        // 排序功能
        headers.forEach(header => {
            header.addEventListener('click', () => {
                const sortBy = header.getAttribute('data-sort');
                const isAscending = header.classList.contains('asc');
                
                // 移除所有排序指示器
                headers.forEach(h => {
                    h.classList.remove('asc', 'desc');
                    h.querySelector('i').className = 'fas fa-sort';
                });
                
                // 设置当前排序方向
                if (isAscending) {
                    header.classList.add('desc');
                    header.querySelector('i').className = 'fas fa-sort-down';
                } else {
                    header.classList.add('asc');
                    header.querySelector('i').className = 'fas fa-sort-up';
                }
                
                // 排序行
                filteredRows.sort((a, b) => {
                    let aValue, bValue;
                    
                    if (sortBy === 'id') {
                        aValue = a.cells[0].getAttribute('title');
                        bValue = b.cells[0].getAttribute('title');
                    } else if (sortBy === 'timestamp') {
                        aValue = a.cells[1].textContent;
                        bValue = b.cells[1].textContent;
                    } else if (sortBy === 'duration' || sortBy === 'total') {
                        const aText = a.cells[sortBy === 'duration' ? 5 : 6].textContent;
                        const bText = b.cells[sortBy === 'duration' ? 5 : 6].textContent;
                        aValue = aText === '-' ? 0 : parseInt(aText.replace(/,/g, '').replace(/[KMG]/g, ''));
                        bValue = bText === '-' ? 0 : parseInt(bText.replace(/,/g, '').replace(/[KMG]/g, ''));
                    } else {
                        aValue = a.cells[sortBy === 'model' ? 2 : (sortBy === 'account' ? 3 : 4)].textContent;
                        bValue = b.cells[sortBy === 'model' ? 2 : (sortBy === 'account' ? 3 : 4)].textContent;
                    }
                    
                    if (aValue < bValue) return isAscending ? -1 : 1;
                    if (aValue > bValue) return isAscending ? 1 : -1;
                    return 0;
                });
                
                // 更新显示
                currentPage = 1;
                initPagination();
            });
        });
        
        // 搜索功能
        const searchInput = document.getElementById('history-search');
        if (searchInput) {
            searchInput.addEventListener('input', function() {
                const searchTerm = this.value.toLowerCase();
                
                filteredRows = rows.filter(row => {
                    const rowText = Array.from(row.cells).map(cell => cell.textContent.toLowerCase()).join(' ');
                    return rowText.includes(searchTerm);
                });
                
                currentPage = 1;
                initPagination();
            });
        }
        
        // 分页控制
        const prevPageBtn = document.getElementById('prev-page');
        const nextPageBtn = document.getElementById('next-page');
        
        if (prevPageBtn) {
            prevPageBtn.addEventListener('click', () => {
                if (currentPage > 1) {
                    currentPage--;
                    initPagination();
                }
            });
        }
        
        if (nextPageBtn) {
            nextPageBtn.addEventListener('click', () => {
                const totalPages = Math.ceil(filteredRows.length / rowsPerPage);
                if (currentPage < totalPages) {
                    currentPage++;
                    initPagination();
                }
            });
        }
        
        // 初始化表格
        initPagination();
    }
    
    // 刷新按钮
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            location.reload();
        });
    }
});

// 初始化图表
function initializeCharts() {
    try {
        // 注册Chart.js插件
        Chart.register(ChartDataLabels);
        
        // 设置全局默认值
        Chart.defaults.font.family = 'Nunito, sans-serif';
        Chart.defaults.color = getComputedStyle(document.documentElement).getPropertyValue('--text-color');
        
        // 每日请求趋势图表
        const dailyChartElement = document.getElementById('dailyChart');
        if (dailyChartElement) {
            const labels = JSON.parse(dailyChartElement.dataset.labels || '[]');
            const values = JSON.parse(dailyChartElement.dataset.values || '[]');
            
            const dailyChart = new Chart(dailyChartElement, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '请求数',
                        data: values,
                        backgroundColor: 'rgba(52, 152, 219, 0.2)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                        pointRadius: 4,
                        tension: 0.3,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            backgroundColor: 'rgba(0, 0, 0, 0.7)',
                            titleFont: {
                                size: 14
                            },
                            bodyFont: {
                                size: 13
                            },
                            padding: 10,
                            displayColors: false
                        },
                        datalabels: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                maxRotation: 45,
                                minRotation: 45
                            }
                        },
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(200, 200, 200, 0.1)'
                            },
                            ticks: {
                                precision: 0
                            }
                        }
                    }
                }
            });
            
            chartInstances['dailyChart'] = dailyChart;
        }
        
        // 模型使用分布图表
        const modelChartElement = document.getElementById('modelChart');
        if (modelChartElement) {
            const labels = JSON.parse(modelChartElement.dataset.labels || '[]');
            const values = JSON.parse(modelChartElement.dataset.values || '[]');
            
            const modelChart = new Chart(modelChartElement, {
                type: 'pie',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '模型使用次数',
                        data: values,
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.5)',
                            'rgba(54, 162, 235, 0.5)',
                            'rgba(255, 206, 86, 0.5)',
                            'rgba(75, 192, 192, 0.5)',
                            'rgba(153, 102, 255, 0.5)',
                            'rgba(255, 159, 64, 0.5)',
                            'rgba(199, 199, 199, 0.5)',
                            'rgba(83, 102, 255, 0.5)',
                            'rgba(40, 159, 64, 0.5)',
                            'rgba(210, 199, 199, 0.5)'
                        ],
                        borderColor: [
                            'rgba(255, 99, 132, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 206, 86, 1)',
                            'rgba(75, 192, 192, 1)',
                            'rgba(153, 102, 255, 1)',
                            'rgba(255, 159, 64, 1)',
                            'rgba(199, 199, 199, 1)',
                            'rgba(83, 102, 255, 1)',
                            'rgba(40, 159, 64, 1)',
                            'rgba(210, 199, 199, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    let label = context.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    label += formatChartNumber(context.parsed);
                                    return label;
                                }
                            }
                        }
                    }
                }
            });
            
            chartInstances['modelChart'] = modelChart;
        }
    } catch (error) {
        console.error('初始化图表失败:', error);
    }
}

// 设置自动刷新功能
function setupAutoRefresh() {
    // 获取已有的刷新进度条元素
    const progressBar = document.getElementById('refresh-progress-bar');
    const countdownElement = document.getElementById('countdown');
    let countdownTimer;
    
    // 倒计时功能
    let countdown = refreshInterval;
    
    function startCountdown() {
        if (countdownTimer) clearInterval(countdownTimer);
        
        countdown = refreshInterval;
        countdownElement.textContent = countdown;
        
        // 重置进度条
        progressBar.style.width = '100%';
        
        if (autoRefreshEnabled) {
            // 设置进度条动画
            progressBar.style.transition = `width ${refreshInterval}s linear`;
            progressBar.style.width = '0%';
            
            countdownTimer = setInterval(function() {
                countdown--;
                if (countdown <= 0) {
                    countdown = refreshInterval;
                    location.reload();
                }
                countdownElement.textContent = countdown;
            }, 1000);
        } else {
            // 暂停进度条动画
            progressBar.style.transition = 'none';
            progressBar.style.width = '0%';
        }
    }
    
    // 立即启动倒计时
    startCountdown();
}

// 设置主题切换
function setupThemeToggle() {
    // 在简化版中，我们移除了主题切换按钮，但保留功能以备将来使用
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
            darkModeEnabled = document.body.classList.contains('dark-mode');
            
            localStorage.setItem('theme', darkModeEnabled ? 'dark' : 'light');
            
            // 更新所有图表的颜色
            updateChartsTheme();
        });
    }
}

// 加载保存的主题
function loadSavedTheme() {
    if (darkModeEnabled) {
        document.body.classList.add('dark-mode');
        const themeToggleBtn = document.querySelector('#theme-toggle-btn i');
        if (themeToggleBtn) {
            themeToggleBtn.classList.remove('fa-moon');
            themeToggleBtn.classList.add('fa-sun');
        }
    }
}

// 更新图表主题
function updateChartsTheme() {
    // 更新所有图表的颜色主题
    Object.values(chartInstances).forEach(chart => {
        // 更新网格线颜色
        if (chart.options.scales && chart.options.scales.y) {
            chart.options.scales.y.grid.color = darkModeEnabled ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
            chart.options.scales.x.grid.color = darkModeEnabled ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
            
            // 更新刻度颜色
            chart.options.scales.y.ticks.color = darkModeEnabled ? '#ddd' : '#666';
            chart.options.scales.x.ticks.color = darkModeEnabled ? '#ddd' : '#666';
        }
        
        // 更新图例颜色
        if (chart.options.plugins && chart.options.plugins.legend) {
            chart.options.plugins.legend.labels.color = darkModeEnabled ? '#ddd' : '#666';
        }
        
        chart.update();
    });
}

// 设置保存统计数据按钮事件
function setupSaveStatsButton() {
    const saveButton = document.querySelector('.save-button');
    if (saveButton) {
        // 添加点击动画效果
        saveButton.addEventListener('click', function() {
            this.classList.add('saving');
            setTimeout(() => {
                this.classList.remove('saving');
            }, 1000);
        });
    }
}

// 添加表格交互功能
function enhanceTableInteraction() {
    // 为请求历史表格添加高亮效果
    const historyRows = document.querySelectorAll('#history-table tbody tr');
    historyRows.forEach(row => {
        row.addEventListener('mouseenter', function() {
            this.classList.add('highlight');
        });
        
        row.addEventListener('mouseleave', function() {
            this.classList.remove('highlight');
        });
    });
}

// 更新页脚信息
function updateFooterInfo() {
    const footer = document.querySelector('.main-footer');
    if (!footer) return;
    
    // 获取当前年份
    const currentYear = new Date().getFullYear();
    
    // 更新版权年份
    const copyrightText = footer.querySelector('p:first-child');
    if (copyrightText) {
        copyrightText.textContent = `© ${currentYear} 2API 统计面板 | 版本 1.0.1`;
    }
}