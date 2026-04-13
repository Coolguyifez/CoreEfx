const CACHE_NAME = 'coreefx-v7';
const OFFLINE_URL = '/'; 

// 1. The list of every file and route you want available offline
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
  '/manifest.json',
  // External assets for offline maps and icons
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
];

// 2. Install Event - Forces the phone to download the full list immediately
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('CoreEfx: Pre-caching all pages and assets');
      // Using mapping with individual catches so one failed file doesn't break the whole cache
      return Promise.allSettled(
        urlsToCache.map(url => {
          return cache.add(url).catch(err => console.warn(`Failed to cache: ${url}`, err));
        })
      );
    })
  );
  // Forces the waiting service worker to become the active service worker
  self.skipWaiting();
});

// 3. Activate Event - Cleans up old caches (v5, v4, etc.) immediately
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys.map(key => {
          if (key !== CACHE_NAME) {
            console.log('CoreEfx: Removing old cache', key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => {
      // Ensures that updates to the service worker take effect immediately
      return self.clients.claim();
    })
  );
});

// 4. Fetch Event - Logic for handling requests offline
self.addEventListener('fetch', event => {
  // Strategy for HTML pages: Network First, then Cache Fallback
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .catch(() => {
          // If offline, check for the specific page (e.g., /chat)
          return caches.match(event.request).then(response => {
            // Fallback to the Splash screen (/) if specific page isn't found
            return response || caches.match(OFFLINE_URL);
          });
        })
    );
  } else {
    // Strategy for Assets (CSS, JS, Images): Cache First, then Network
    event.respondWith(
      caches.match(event.request).then(response => {
        return response || fetch(event.request).then(networkResponse => {
          // Optional: Cache new images or files found while browsing
          return caches.open(CACHE_NAME).then(cache => {
            cache.put(event.request, networkResponse.clone());
            return networkResponse;
          });
        }).catch(() => {
            // If both fail, just return nothing
            return null;
        });
      })
    );
  }
});
