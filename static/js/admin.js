document.addEventListener('DOMContentLoaded', function() {
    // Load initial data
    loadDashboardData();
    
    // Refresh data button
    document.getElementById('refresh-data').addEventListener('click', function() {
        loadDashboardData();
    });
    
    // Export data button
    document.getElementById('export-data').addEventListener('click', function() {
        exportData();
    });
    
    // Sidebar navigation
    const navLinks = document.querySelectorAll('.sidebar .nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // Handle navigation (you can add specific functionality for each section)
            const section = this.getAttribute('href').substring(1);
            handleNavigation(section);
        });
    });
});

function loadDashboardData() {
    // Load statistics
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            // Update statistics cards
            document.getElementById('total-predictions').textContent = data.total_count;
            document.getElementById('success-rate').textContent = 
                (data.success_rate * 100).toFixed(1) + '%';
            document.getElementById('active-users').textContent = data.active_users;
            document.getElementById('system-status').textContent = data.system_status;
            
            // Update system status indicator
            const statusIndicator = document.createElement('span');
            statusIndicator.className = 'status-indicator';
            statusIndicator.classList.add('status-' + data.system_status.toLowerCase());
            document.getElementById('system-status').prepend(statusIndicator);
        })
        .catch(error => console.error('Error loading stats:', error));
    
    // Load recent predictions
    fetch('/api/predictions')
        .then(response => response.json())
        .then(data => {
            const tableBody = document.getElementById('predictions-table');
            tableBody.innerHTML = '';
            
            data.forEach(prediction => {
                const row = document.createElement('tr');
                row.className = 'fade-in';
                
                row.innerHTML = `
                    <td>${prediction.id}</td>
                    <td>${new Date(prediction.timestamp).toLocaleString()}</td>
                    <td>${prediction.blood_group}</td>
                    <td>${(prediction.confidence * 100).toFixed(2)}%</td>
                    <td>
                        <span class="badge bg-${getStatusColor(prediction.status)}">
                            ${prediction.status}
                        </span>
                    </td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary" onclick="viewDetails(${prediction.id})">
                            View
                        </button>
                        <button class="btn btn-sm btn-outline-danger" onclick="deletePrediction(${prediction.id})">
                            Delete
                        </button>
                    </td>
                `;
                
                tableBody.appendChild(row);
            });
        })
        .catch(error => console.error('Error loading predictions:', error));
    
    // Load system logs
    fetch('/api/logs')
        .then(response => response.json())
        .then(data => {
            const logContainer = document.getElementById('system-logs');
            logContainer.innerHTML = '';
            
            data.forEach(log => {
                const logEntry = document.createElement('div');
                logEntry.className = `log-entry ${log.level.toLowerCase()}`;
                logEntry.innerHTML = `
                    <span class="timestamp">${new Date(log.timestamp).toLocaleString()}</span>
                    <span class="level">${log.level}</span>
                    <span class="message">${log.message}</span>
                `;
                logContainer.appendChild(logEntry);
            });
            
            // Scroll to bottom
            logContainer.scrollTop = logContainer.scrollHeight;
        })
        .catch(error => console.error('Error loading logs:', error));
}

function getStatusColor(status) {
    switch(status.toLowerCase()) {
        case 'success':
            return 'success';
        case 'pending':
            return 'warning';
        case 'failed':
            return 'danger';
        default:
            return 'secondary';
    }
}

function handleNavigation(section) {
    // Handle navigation to different sections
    console.log('Navigating to:', section);
    // Add specific functionality for each section here
}

function viewDetails(id) {
    // View detailed information about a prediction
    fetch(`/api/predictions/${id}`)
        .then(response => response.json())
        .then(data => {
            // Show modal with details
            showModal('Prediction Details', `
                <div class="prediction-details">
                    <p><strong>ID:</strong> ${data.id}</p>
                    <p><strong>Timestamp:</strong> ${new Date(data.timestamp).toLocaleString()}</p>
                    <p><strong>Blood Group:</strong> ${data.blood_group}</p>
                    <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                    <p><strong>Status:</strong> ${data.status}</p>
                    <p><strong>Image:</strong></p>
                    <img src="${data.image_url}" class="img-fluid" alt="Fingerprint Image">
                </div>
            `);
        })
        .catch(error => console.error('Error loading prediction details:', error));
}

function deletePrediction(id) {
    if (confirm('Are you sure you want to delete this prediction?')) {
        fetch(`/api/predictions/${id}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                loadDashboardData(); // Reload the data
                showAlert('success', 'Prediction deleted successfully');
            } else {
                showAlert('error', 'Failed to delete prediction');
            }
        })
        .catch(error => {
            console.error('Error deleting prediction:', error);
            showAlert('error', 'An error occurred while deleting the prediction');
        });
    }
}

function exportData() {
    fetch('/api/export')
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'predictions.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        })
        .catch(error => {
            console.error('Error exporting data:', error);
            showAlert('error', 'Failed to export data');
        });
}

function showModal(title, content) {
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${title}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
    
    modal.addEventListener('hidden.bs.modal', function() {
        document.body.removeChild(modal);
    });
}

function showAlert(type, message) {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.querySelector('main').insertBefore(alert, document.querySelector('main').firstChild);
    
    setTimeout(() => {
        alert.remove();
    }, 5000);
} 