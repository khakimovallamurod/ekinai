document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultsDiv = document.getElementById('results');
    const downloadBtn = document.getElementById('downloadBtn');
    const statsBtn = document.getElementById('statsBtn');
    let currentResult = null;

    // Range inputlar uchun qiymat ko'rsatish
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        const valueSpan = input.nextElementSibling;
        
        input.addEventListener('input', function() {
            const value = parseFloat(this.value);
            if (this.name === 'microorg') {
                valueSpan.textContent = value.toExponential(2);
            } else {
                valueSpan.textContent = value.toFixed(2);
            }
        });
        
        const value = parseFloat(input.value);
        if (input.name === 'microorg') {
            valueSpan.textContent = value.toExponential(2);
        } else {
            valueSpan.textContent = value.toFixed(2);
        }
    });

    // Forma yuborish
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        showSpinner(true);
        
        const formData = new FormData(form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (result.success) {
                currentResult = result;
                displayResults(result);
                showNotification('Bashorat muvaffaqiyatli bajarildi!', 'success');
            } else {
                showNotification('Xatolik: ' + result.error, 'error');
            }
        } catch (error) {
            showNotification('Server bilan aloqa xatosi', 'error');
            console.error('Error:', error);
        } finally {
            showSpinner(false);
        }
    });

    // Natijalarni ko'rsatish
    function displayResults(result) {
        resultsDiv.style.display = 'block';
        
        const cropEmojis = {
            "Bug'doy": "🌾",
            "Kartoshka": "🥔",
            "Loviya": "🫘",
            "Qalampir": "🌶️",
            "Makkajo'xori": "🌽",
            "Sabzi": "🥕",
            "Paxta": "☁️"
        };
        
        const emoji = cropEmojis[result.best_crop] || "🌾";
        document.getElementById('bestCrop').innerHTML = `
            <div>Eng mos ekin:</div>
            <div style="color: #15803d; margin-top: 10px;">
                ${result.best_crop} (${result.best_prob.toFixed(1)}%) ${emoji}
            </div>
        `;
        
        // Grafikni ko'rsatish
        document.getElementById('graphContainer').innerHTML = `
            <img src="data:image/png;base64,${result.graph}" style="max-width: 100%; height: auto;">
        `;
        
        // Taqqoslash grafigi
        document.getElementById('comparisonContainer').innerHTML = `
            <h3 style="text-align: center; color: #2d5016; margin: 20px 0;">
                ${result.best_crop}ga muhim xususiyatlar
            </h3>
            <img src="data:image/png;base64,${result.comparison_graph}" style="max-width: 100%; height: auto;">
        `;
        
        // Jadval
        displayProbTable(result.probabilities);
        
        // Ekin rasmi
        displayCropImage(result.best_crop);
        
        resultsDiv.scrollIntoView({ behavior: 'smooth' });
    }

    // Mosliklar jadvali
    function displayProbTable(probabilities) {
        let tableHTML = `
            <table class="prob-table">
                <thead>
                    <tr>
                        <th>Ekin</th>
                        <th>Moslik (%)</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        probabilities.forEach(item => {
            tableHTML += `
                <tr>
                    <td>${item.Ekin}</td>
                    <td>${item['Moslik (%)'].toFixed(1)}%</td>
                </tr>
            `;
        });
        
        tableHTML += '</tbody></table>';
        document.getElementById('probTable').innerHTML = tableHTML;
    }

    // Ekin rasmi
    function displayCropImage(crop) {
        const cropImages = {
            "Bug'doy": "https://i0.wp.com/razzanj.com/wp-content/uploads/2016/07/nature-landscape-nature-landscape-hd-image-download-wheat-farm-hd-wallpaper-notebook-background-wheat-farmers-wheat-farming-process-wheat-farming-in-kenya.jpg?ssl=1",
            "Kartoshka": "https://www.isaaa.org/kc/cropbiotechupdate/files/images/3172020111359PM.jpg",
            "Loviya": "https://cdn.britannica.com/24/122524-050-4593E7D1/Green-beans.jpg",
            "Qalampir": "https://img.freepik.com/premium-photo/red-chili-chilli-peppers-plant-garden-agricultural-plantation-farm-countryside-nonthaburi-thailand_258052-6029.jpg",
            "Makkajo'xori": "https://www.aces.edu/wp-content/uploads/2018/08/shutterstock_-Zeljko-Radojko_field-corn.jpg",
            "Sabzi": "https://ogden_images.s3.amazonaws.com/www.motherearthnews.com/images/2022/02/11110505/growing-carrots.jpg",
            "Paxta": "https://cdn.pixabay.com/photo/2014/02/13/12/56/cotton-crop-265312_1280.jpg"
        };
        
        if (cropImages[crop]) {
            document.getElementById('cropImage').innerHTML = `
                <h3 style="color: #2d5016; margin-bottom: 15px;">${crop} ekini rasmi</h3>
                <img src="${cropImages[crop]}" alt="${crop}" class="crop-image">
            `;
        }
    }

    // CSV formatida yuklab olish
    downloadBtn.addEventListener('click', function() {
        if (!currentResult) return;
        
        let csvContent = "data:text/csv;charset=utf-8,";
        
        // Kiritilgan qiymatlar
        csvContent += "KIRITILGAN QIYMATLAR\n";
        csvContent += "Xususiyat,Qiymat\n";
        for (let [key, value] of Object.entries(currentResult.input_dict)) {
            csvContent += `${key},${value}\n`;
        }
        
        csvContent += "\n\nEKIN MOSLIKLARI\n";
        csvContent += "Ekin,Moslik (%)\n";
        currentResult.probabilities.forEach(item => {
            csvContent += `${item.Ekin},${item['Moslik (%)'].toFixed(2)}\n`;
        });
        
        csvContent += "\n\nENG MOS EKIN\n";
        csvContent += `${currentResult.best_crop} (${currentResult.best_prob.toFixed(1)}%)\n`;
        
        // Download
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "ekin_natijasi.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showNotification('Fayl muvaffaqiyatli yuklandi!', 'success');
    });

    // Statistikani ko'rsatish
    if (statsBtn) {
        statsBtn.addEventListener('click', async function() {
            showSpinner(true);
            
            try {
                const response = await fetch('/stats');
                const result = await response.json();
                
                if (result.success) {
                    displayStats(result);
                } else {
                    showNotification('Xatolik: ' + result.error, 'error');
                }
            } catch (error) {
                showNotification('Server bilan aloqa xatosi', 'error');
            } finally {
                showSpinner(false);
            }
        });
    }

    function displayStats(stats) {
        const modal = document.getElementById('statsModal');
        
        document.getElementById('statsContent').innerHTML = `
            <h3>Model aniqligi</h3>
            <div class="metrics">
                <div class="metric">
                    <span>Train:</span>
                    <strong>${(stats.metrics.train_acc * 100).toFixed(2)}%</strong>
                </div>
                <div class="metric">
                    <span>Test:</span>
                    <strong>${(stats.metrics.test_acc * 100).toFixed(2)}%</strong>
                </div>
                <div class="metric">
                    <span>CV:</span>
                    <strong>${(stats.metrics.cv_acc * 100).toFixed(2)}%</strong>
                </div>
            </div>
            <h3 style="margin-top: 30px;">Siniflar Taqsimoti</h3>
            <img src="data:image/png;base64,${stats.class_graph}" style="max-width: 100%; height: auto;">
        `;
        
        modal.style.display = 'block';
    }

    // Modal yopish
    const modal = document.getElementById('statsModal');
    const closeBtn = document.querySelector('.close-modal');
    
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            modal.style.display = 'none';
        });
    }
    
    window.addEventListener('click', function(event) {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    });

    // Yordamchi funksiyalar
    function showSpinner(show) {
        let spinner = document.querySelector('.spinner');
        if (!spinner) {
            spinner = document.createElement('div');
            spinner.className = 'spinner';
            form.appendChild(spinner);
        }
        spinner.classList.toggle('active', show);
    }

    function showNotification(message, type) {
        let notification = document.querySelector('.notification');
        if (!notification) {
            notification = document.createElement('div');
            notification.className = 'notification';
            document.body.appendChild(notification);
        }
        
        notification.textContent = message;
        notification.className = `notification ${type} show`;
        
        setTimeout(() => {
            notification.classList.remove('show');
        }, 3000);
    }
});