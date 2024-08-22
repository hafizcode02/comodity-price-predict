// Area Datetime
$(document).ready(function () {
    var baseOptions = {
        chart: {
            type: 'area',
            height: 288,
            toolbar: {
                show: false,
            },
        },
        colors: ['#59c4bc'],
        dataLabels: {
            enabled: false
        },

        series: [{
            name: '',
            data: [],
        },

        ],
        markers: {
            size: 0,
            style: 'hollow',
        },
        xaxis: {
            type: 'datetime',
            tickAmount: 6,
            show: false,
        },
        tooltip: {
            x: {
                format: 'dd MMM yyyy'
            }
        },
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.7,
                opacityTo: 0.9,
                stops: [0, 100]
            }
        },
        stroke: {
            show: true,
            curve: 'smooth',
            width: 2,
        },
        grid: {
            yaxis: {
                lines: {
                    show: false,
                }
            },
        },
    }

    // Function to find min and max dates in your data
    function getMinMaxDates(data) {
        const timestamps = data.map(item => item[0]); // Assuming data is like [[timestamp, value], ...]
        return {
            minDate: Math.min(...timestamps),
            maxDate: Math.max(...timestamps)
        };
    }

    var resetCssClasses = function (activeEl) {
        var els = document.querySelectorAll("button");
        Array.prototype.forEach.call(els, function (el) {
            el.classList.remove('active');
        });

        activeEl.target.classList.add('active')
    }

    function setupRangeButtons(chartSelector, chart, divIdSelector, minMaxDate) {
        // Construct button selectors based on the chartSelector
        document.querySelector(divIdSelector + " #one_month").addEventListener('click', function (e) {
            resetCssClasses(e)
            chart.updateOptions({
                xaxis: {
                    min: new Date(minMaxDate.maxDate - 30 * 24 * 60 * 60 * 1000).getTime(), // 30 days before the max date
                    max: minMaxDate.maxDate
                }
            });
        });

        document.querySelector(divIdSelector + " #six_months").addEventListener('click', function (e) {
            resetCssClasses(e)
            chart.updateOptions({
                xaxis: {
                    min: new Date(minMaxDate.maxDate - 6 * 30 * 24 * 60 * 60 * 1000).getTime(), // 6 months before the max date
                    max: minMaxDate.maxDate
                }
            });
        });

        document.querySelector(divIdSelector + " #one_year").addEventListener('click', function (e) {
            resetCssClasses(e)
            chart.updateOptions({
                xaxis: {
                    min: new Date(minMaxDate.maxDate - 365 * 24 * 60 * 60 * 1000).getTime(), // 1 year before the max date
                    max: minMaxDate.maxDate
                }
            });
        });

        document.querySelector(divIdSelector + " #ytd").addEventListener('click', function (e) {
            resetCssClasses(e)
            const startOfYear = new Date(new Date(minMaxDate.maxDate).getFullYear(), 0, 1).getTime(); // Start of the current year
            chart.updateOptions({
                xaxis: {
                    min: startOfYear,
                    max: minMaxDate.maxDate
                }
            });
        });

        document.querySelector(divIdSelector + " #all").addEventListener('click', function (e) {
            resetCssClasses(e)
            chart.updateOptions({
                xaxis: {
                    min: minMaxDate.minDate,
                    max: minMaxDate.maxDate
                }
            });
        });
    }

    // Function to clone base options and customize
    function createChartOptions(baseOptions, seriesName, seriesData, color, chartSelector, minMaxDate, divIdSelector) {
        let options = JSON.parse(JSON.stringify(baseOptions));

        // Round the data values to remove decimals
        seriesData = seriesData.map(([timestamp, value]) => [timestamp, Math.round(value)]);

        options.series[0].name = seriesName;
        options.series[0].data = seriesData;
        options.colors = [color];

        var chart = new ApexCharts(document.querySelector(chartSelector), options);
        chart.render();

        setupRangeButtons(chartSelector, chart, divIdSelector, minMaxDate);
    }

    fetch('http://localhost:2024/comodity-data/cabai-merah-besar')
        .then(response => response.json())
        .then(data => {
            let minMaxDate = getMinMaxDates(data.data);
            createChartOptions(baseOptions, "Cabai Merah Besar", data.data, '#59c4bc', '#apex-cabai-merah-besar', minMaxDate, '#cabai-merah-besar');
        })
        .catch(error => console.error('Error fetching data for chart cabai-merah-besar:', error));

    fetch('http://localhost:2024/comodity-data/cabai-merah-keriting')
        .then(response => response.json())
        .then(data => {
            let minMaxDate = getMinMaxDates(data.data);
            createChartOptions(baseOptions, "Cabai Merah Besar", data.data, '#59c4bc', '#apex-cabai-merah-keriting', minMaxDate, '#cabai-merah-keriting');
        })
        .catch(error => console.error('Error fetching data for chart cabai-merah-keriting:', error));

    fetch('http://localhost:2024/comodity-data/cabai-rawit-merah')
        .then(response => response.json())
        .then(data => {
            let minMaxDate = getMinMaxDates(data.data);
            createChartOptions(baseOptions, "Cabai Merah Besar", data.data, '#59c4bc', '#apex-cabai-rawit-merah', minMaxDate, '#cabai-rawit-merah');
        })
        .catch(error => console.error('Error fetching data for chart cabai-rawit-merah:', error));

    fetch('http://localhost:2024/comodity-data/cabai-rawit-hijau')
        .then(response => response.json())
        .then(data => {
            let minMaxDate = getMinMaxDates(data.data);
            createChartOptions(baseOptions, "Cabai Merah Besar", data.data, '#59c4bc', '#apex-cabai-rawit-hijau', minMaxDate, '#cabai-rawit-hijau');
        })
        .catch(error => console.error('Error fetching data for chart cabai-rawit-hijau:', error));

    fetch('http://localhost:2024/comodity-data/bawang-merah')
        .then(response => response.json())
        .then(data => {
            let minMaxDate = getMinMaxDates(data.data);
            createChartOptions(baseOptions, "Cabai Merah Besar", data.data, '#59c4bc', '#apex-bawang-merah', minMaxDate, '#bawang-merah');
        })
        .catch(error => console.error('Error fetching data for chart bawang-merah:', error));
});