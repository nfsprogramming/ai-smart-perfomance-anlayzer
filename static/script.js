document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    // UI Loading State
    const btn = document.querySelector('.btn-primary');
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.loading-spinner');

    btnText.style.display = 'none';
    spinner.classList.remove('hidden');
    btn.disabled = true;

    // Collect Data
    const data = {
        marks: document.getElementById('marks').value,
        attendance: document.getElementById('attendance').value,
        sleep_hours: document.getElementById('sleep_hours').value,
        screen_time: document.getElementById('screen_time').value,
        assignment_delay: document.getElementById('assignment_delay').value,
        feedback: document.getElementById('feedback').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (response.ok) {
            displayResults(result);
        } else {
            alert('Error: ' + (result.error || 'Something went wrong'));
        }

    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to the server.');
    } finally {
        // Reset UI
        btnText.style.display = 'block';
        spinner.classList.add('hidden');
        btn.disabled = false;
    }
});

function displayResults(data) {
    const resultSection = document.getElementById('resultSection');
    resultSection.classList.remove('hidden');

    // Smooth scroll to result
    resultSection.scrollIntoView({ behavior: 'smooth' });

    // 1. Risk Badge
    const riskBadge = document.getElementById('riskBadge');
    const riskIndex = data.prediction.risk_index;

    riskBadge.className = 'risk-badge'; // Reset
    if (riskIndex === 0) {
        riskBadge.classList.add('risk-low');
        riskBadge.textContent = 'Low Risk 🟢';
    } else if (riskIndex === 1) {
        riskBadge.classList.add('risk-med');
        riskBadge.textContent = 'Medium Risk 🟡';
    } else {
        riskBadge.classList.add('risk-high');
        riskBadge.textContent = 'High Risk 🔴';
    }

    // 2. Probabilities
    const probs = data.prediction.final_probs; // [low, med, high]
    document.getElementById('probLow').style.width = (probs[0] * 100) + '%';
    document.getElementById('probMed').style.width = (probs[1] * 100) + '%';
    document.getElementById('probHigh').style.width = (probs[2] * 100) + '%';

    // 3. Recommendations
    const recList = document.getElementById('recommendationsList');
    recList.innerHTML = '';
    data.recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        recList.appendChild(li);
    });

    // 4. Feature Importance
    const featContainer = document.getElementById('featureImportance');
    featContainer.innerHTML = '';
    const importances = data.feature_importance;

    // Convert to array and sort
    const sortedFeatures = Object.entries(importances).sort(([, a], [, b]) => b - a);

    sortedFeatures.forEach(([key, value]) => {
        const div = document.createElement('div');
        div.className = 'feature-tag';
        // Format key (e.g., sleep_hours -> Sleep Hours)
        const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        div.innerHTML = `${label} <span>${(value * 100).toFixed(1)}%</span>`;
        featContainer.appendChild(div);
    });
}
