const CACHE_NAME = 'coreefx-cache-v3'; // Bumped version
const OFFLINE_URL = '/'; // This is the page users see when offline

const urlsToCache = [
  '/',
  '/static/style.css', 
  '/static/images/brain.png',
  '/login'
];

// 1. Install: Pre-cache the "Offline" landing page
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('Cache Pre-filled');
      return cache.addAll(urlsToCache);
    })
  );
  self.skipWaiting(); // Force the new service worker to take over immediately
});

// 2. Activate: Clean up old cache versions so they don't take up space
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

// 3. Fetch: The Offline Strategy
self.addEventListener('fetch', event => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .catch(() => {
          // If the network fails, always return the cached Splash/Home page
          // This prevents the "No Internet" dinosaur/browser screen
          return caches.match(OFFLINE_URL);
        })
    );
  } else {
    // For images, CSS, and JS: try Cache first, then Network
    event.respondWith(
      caches.match(event.request).then(response => {
        return response || fetch(event.request);
      })
    );
  }
});
