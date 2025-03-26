// Main Interface JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const loadingElement = document.getElementById('loading');
    const resultElement = document.getElementById('result');
    const bloodGroupElement = document.getElementById('blood-group');
    const confidenceElement = document.getElementById('confidence');
    const confidenceBar = document.getElementById('confidence-bar');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const fileInput = document.getElementById('fingerprint-image');

    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get file
            const file = fileInput.files[0];
            if (!file) {
                alert('Please select a file');
                return;
            }
            
            // Display preview
            previewContainer.style.display = 'block';
            previewImage.src = URL.createObjectURL(file);
            
            // Show loading, hide result
            loadingElement.style.display = 'block';
            resultElement.style.display = 'none';
            
            // Create form data
            const formData = new FormData();
            formData.append('file', file);
            
            // Send request
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Hide loading
                loadingElement.style.display = 'none';
                
                // Show result
                resultElement.style.display = 'block';
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Display blood group and confidence
                bloodGroupElement.textContent = data.blood_group;
                const confidencePercent = (data.confidence * 100).toFixed(2);
                confidenceElement.textContent = confidencePercent;
                confidenceBar.style.width = confidencePercent + '%';
                
                // Set color based on confidence
                if (confidencePercent < 50) {
                    confidenceBar.style.backgroundColor = '#dc3545'; // red
                } else if (confidencePercent < 75) {
                    confidenceBar.style.backgroundColor = '#ffc107'; // yellow
                } else {
                    confidenceBar.style.backgroundColor = '#28a745'; // green
                }
            })
            .catch(error => {
                loadingElement.style.display = 'none';
                alert('Error: ' + error.message);
            });
        });
    }

    // Handle image preview on file select
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const file = fileInput.files[0];
            if (file) {
                previewContainer.style.display = 'block';
                previewImage.src = URL.createObjectURL(file);
            }
        });
    }
});

// Admin Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts
    let bloodGroupChart = null;
    let confidenceChart = null;
    
    // Load initial data
    loadDashboardData();
    
    // Refresh data button
    const refreshButton = document.getElementById('refresh-data');
    if (refreshButton) {
        refreshButton.addEventListener('click', loadDashboardData);
    }
    
    // Train model form
    const trainForm = document.getElementById('train-model-form');
    if (trainForm) {
        trainForm.addEventListener('submit', handleModelTraining);
    }
    
    // Download model button
    const downloadButton = document.getElementById('download-model');
    if (downloadButton) {
        downloadButton.addEventListener('click', downloadModel);
    }
});

// Load dashboard data
async function loadDashboardData() {
    try {
        // Load statistics
        const statsResponse = await fetch('/api/stats');
        const statsData = await statsResponse.json();
        
        // Update statistics cards
        document.getElementById('total-predictions').textContent = statsData.total_count;
        document.getElementById('avg-confidence').textContent = 
            (statsData.avg_confidence * 100).toFixed(2) + '%';
        document.getElementById('most-common').textContent = statsData.most_common;
        document.getElementById('new-today').textContent = statsData.new_today;
        
        // Update charts
        updateBloodGroupChart(statsData.blood_group_counts);
        updateConfidenceChart(statsData.confidence_distribution);
        
        // Load recent predictions
        const predictionsResponse = await fetch('/api/predictions');
        const predictionsData = await predictionsResponse.json();
        
        // Update predictions table
        const tableBody = document.getElementById('predictions-table-body');
        tableBody.innerHTML = '';
        
        predictionsData.forEach(prediction => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${prediction.id}</td>
                <td>${prediction.filename}</td>
                <td>${prediction.blood_group}</td>
                <td>${(prediction.confidence * 100).toFixed(2)}%</td>
                <td>${new Date(prediction.timestamp).toLocaleString()}</td>
            `;
            tableBody.appendChild(row);
        });
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showAlert('error', 'Failed to load dashboard data');
    }
}

// Update blood group chart
function updateBloodGroupChart(data) {
    const ctx = document.getElementById('blood-group-chart').getContext('2d');
    
    const labels = Object.keys(data);
    const values = Object.values(data);
    
    if (bloodGroupChart) {
        bloodGroupChart.destroy();
    }
    
    bloodGroupChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: [
                    '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', 
                    '#9966FF', '#FF9F40', '#8AC249', '#EA526F'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right'
                },
                title: {
                    display: true,
                    text: 'Blood Group Distribution'
                }
            }
        }
    });
}

// Update confidence chart
function updateConfidenceChart(data) {
    const ctx = document.getElementById('confidence-chart').getContext('2d');
    
    const labels = ['0-25%', '25-50%', '50-75%', '75-100%'];
    
    if (confidenceChart) {
        confidenceChart.destroy();
    }
    
    confidenceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence Distribution',
                data: [
                    data['0-25'] || 0,
                    data['25-50'] || 0,
                    data['50-75'] || 0,
                    data['75-100'] || 0
                ],
                backgroundColor: ['#FF6384', '#FFCE56', '#36A2EB', '#4BC0C0']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Prediction Confidence Distribution'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Predictions'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Confidence Range'
                    }
                }
            }
        }
    });
}

// Handle model training
async function handleModelTraining(e) {
    e.preventDefault();
    
    const datasetPath = document.getElementById('dataset-path').value;
    const epochs = document.getElementById('epochs').value;
    const batchSize = document.getElementById('batch-size').value;
    
    const submitBtn = this.querySelector('button[type="submit"]');
    const originalBtnText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Training... Please wait';
    
    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_path: datasetPath,
                epochs: parseInt(epochs),
                batch_size: parseInt(batchSize)
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showAlert('error', data.error);
        } else {
            showAlert('success', 'Model trained successfully!');
            
            // Update model info
            document.getElementById('val-accuracy').textContent = 
                (data.history.val_accuracy * 100).toFixed(2) + '%';
            document.getElementById('last-trained').textContent = new Date().toLocaleString();
            
            // Reload dashboard data
            loadDashboardData();
        }
    } catch (error) {
        showAlert('error', 'Failed to train model: ' + error.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = originalBtnText;
    }
}

// Download model
async function downloadModel() {
    try {
        const response = await fetch('/download-model');
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'blood_group_model.h5';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    } catch (error) {
        showAlert('error', 'Failed to download model: ' + error.message);
    }
}

// Show alert message
function showAlert(type, message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
} 