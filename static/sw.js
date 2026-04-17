const CACHE_NAME = 'coreefx-v9'; // Incremented version
const OFFLINE_URL = '/'; 

// 1. Assets to pre-cache
const urlsToCache = [
  '/', 
  '/login',
  '/signup',
  '/home',
  '/chat',
  '/metrics',
  '/privacy_policy',
  '/terms_of_service',
  '/static/style.css', 
  '/static/images/brain.png',
  '/manifest.json',
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
];

// 2. Install Event
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('CoreEfx: Pre-caching all pages and assets');
      return Promise.allSettled(
        urlsToCache.map(url => {
          return cache.add(url).catch(err => console.warn(`Failed to cache: ${url}`, err));
        })
      );
    })
  );
  self.skipWaiting();
});

// 3. Activate Event
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
      return self.clients.claim();
    })
  );
});

// 4. Fetch Event
self.addEventListener('fetch', event => {
  // FIX 1: The Cache API only supports 'GET'. 
  // We must ignore POST requests (like symptom submissions or feedback).
  if (event.request.method !== 'GET') {
    return; 
  }

  // Strategy for HTML pages: Network First, then Cache Fallback
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .catch(() => {
          return caches.match(event.request).then(response => {
            return response || caches.match(OFFLINE_URL);
          });
        })
    );
  } else {
    // Strategy for Assets (CSS, JS, Images): Cache First, then Network
    event.respondWith(
      caches.match(event.request).then(response => {
        return response || fetch(event.request).then(networkResponse => {
          
          // Check for valid status. 
          // Status 206 (Partial Content) is used for streaming audio (MP3s) 
          // and cannot be stored in the Cache API.
          if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
            return networkResponse;
          }

          return caches.open(CACHE_NAME).then(cache => {
            cache.put(event.request, networkResponse.clone());
            return networkResponse;
          });
        }).catch(() => {
            return null;
        });
      })
    );
  }
});
