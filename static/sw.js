const CACHE_NAME = 'coreefx-v5';
const OFFLINE_URL = '/'; 

const urlsToCache = [
  '/', 
  '/login',
  '/signup',
  '/home',
  '/chat',
  '/privacy_policy',
  '/terms_of_service',
  '/static/style.css', 
  '/static/images/brain.png',
  '/manifest.json'
];

// 1. Install - Try to cache everything
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('CoreEfx: Pre-caching assets');
      // Promise.allSettled is safer than addAll
      return Promise.allSettled(
        urlsToCache.map(url => {
          return cache.add(url).catch(err => console.warn(`Failed to cache: ${url}`, err));
        })
      );
    })
  );
  self.skipWaiting();
});

// 2. Activate - Cleanup old versions
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.map(key => {
        if (key !== CACHE_NAME) return caches.delete(key);
      })
    ))
  );
});

// 3. Fetch - Offline Logic
self.addEventListener('fetch', event => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .catch(() => {
          // If network fails, try to find the SPECIFIC page requested (e.g., /chat)
          return caches.match(event.request).then(response => {
            // If the specific page isn't in cache, show the Splash Screen (OFFLINE_URL)
            return response || caches.match(OFFLINE_URL);
          });
        })
    );
  } else {
    // For images/CSS: Cache first, then Network
    event.respondWith(
      caches.match(event.request).then(response => {
        return response || fetch(event.request);
      })
    );
  }
});
