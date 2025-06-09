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

// Sample classifications for demo - Delete later on
const sampleClassifications = [
  { primary: 'Weeb',  secondary: 'Slight Furry', weebScore: 0.018, furryScore: 0.009, primaryClass: 'weeb', secondaryClass: 'furry' },
  { primary: 'Furry', secondary: 'None',          weebScore: 0.005, furryScore: 0.021, primaryClass: 'furry', secondaryClass: 'normie' },
  { primary: 'Normie',secondary: 'None',          weebScore: 0.002, furryScore: 0.001, primaryClass: 'normie', secondaryClass: 'none' },
  { primary: 'Weeb',  secondary: 'None',          weebScore: 0.015, furryScore: 0.003, primaryClass: 'weeb', secondaryClass: 'none' }
];

// Event Listeners
checkBtn.addEventListener('click', () => {
  const username = usernameInput.value.trim();
  if (!username) {
    alert('Please enter a Bluesky username (e.g., username.bsky.social)');
    return;
  }
  loadingIndicator.style.display = 'block';
  resultContainer.style.display = 'none';

  // Replace with API
  setTimeout(() => {
    loadingIndicator.style.display = 'none';
    resultContainer.style.display = 'block';
    statsContainer.style.display = 'none';
    resultUsername.textContent = `@${username}`;
    const randomResult = sampleClassifications[Math.floor(Math.random() * sampleClassifications.length)];
    primaryResult.textContent   = randomResult.primary;
    secondaryResult.textContent = randomResult.secondary;
    weebScore.textContent       = randomResult.weebScore.toFixed(3);
    furryScore.textContent      = randomResult.furryScore.toFixed(3);
    primaryResult.className     = 'classification-value ' + randomResult.primaryClass;
    secondaryResult.className   = 'classification-value ' + randomResult.secondaryClass;
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, 2000);
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