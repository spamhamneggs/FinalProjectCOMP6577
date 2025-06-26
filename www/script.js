// DOM Elements
const checkBtn       = document.getElementById('check-btn');
const newSearchBtn   = document.getElementById('new-search-btn');
const usernameInput  = document.getElementById('username');
const resultContainer= document.getElementById('result-container');
const statsContainer = document.getElementById('stats');
const loadingIndicator = document.getElementById('loading');
const resultUsername = document.getElementById('result-username');
const primaryResult  = document.getElementById('primary-result');
const secondaryResult= document.getElementById('secondary-result');
const weebScore      = document.getElementById('weeb-score');
const furryScore     = document.getElementById('furry-score');
const userCount      = document.getElementById('user-count');
const postCount      = document.getElementById('post-count');

// Event Listeners
checkBtn.addEventListener('click', async () => {
  const username = usernameInput.value.trim();
  if (!username) {
    alert('Please enter a Bluesky username (e.g., username.bsky.social)');
    return;
  }
  loadingIndicator.style.display = 'block';
  resultContainer.style.display = 'none';
  statsContainer.style.display = 'none';

  try {
    const response = await fetch('/api/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username })
    });
    const data = await response.json();
    loadingIndicator.style.display = 'none';
    if (!response.ok || data.error) {
      alert(data.error || 'Classification failed.');
      statsContainer.style.display = 'flex';
      return;
    }
    resultContainer.style.display = 'block';
    resultUsername.textContent = `@${username}`;
    primaryResult.textContent   = data.primary_classification || 'Unknown';
    secondaryResult.textContent = data.secondary_classification || 'Unknown';
    weebScore.textContent       = (data.average_weeb_score * 100 ?? 0).toFixed(3);
    furryScore.textContent      = (data.average_furry_score * 100 ?? 0).toFixed(3);
    primaryResult.className     = 'classification-value ' + (data.primary_classification?.toLowerCase().replace(/\s/g, '-') || 'unknown');
    secondaryResult.className   = 'classification-value ' + (data.secondary_classification?.toLowerCase().replace(/\s/g, '-') || 'unknown');
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
  } catch (err) {
    loadingIndicator.style.display = 'none';
    alert('Error connecting to the classifier API.');
    statsContainer.style.display = 'flex';
  }
});

newSearchBtn.addEventListener('click', () => {
  usernameInput.value = '';
  resultContainer.style.display = 'none';
  statsContainer.style.display = 'flex';
  loadingIndicator.style.display = 'none';
  usernameInput.focus();
  document.querySelector('.classifier-container').scrollIntoView({ behavior: 'smooth' });
});

// Initialize the page
document.addEventListener('DOMContentLoaded', initStats);