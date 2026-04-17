const CACHE_NAME = 'coreefx-cache-v3'; // Changed version to force an update 
const urlsToCache = [
  '/',
  '/static/style.css', 
  '/static/images/brain.png',
  '/login' // Add your login page to the initial cache
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(urlsToCache))
  );
});

// --- NEW NETWORK-FIRST STRATEGY ---
self.addEventListener('fetch', event => {
  if (event.request.mode === 'navigate') {
    // For page changes, try the network (the server) first
    event.respondWith(
      fetch(event.request)
        .catch(() => {
          // If offline, then look in the cache
          return caches.match(event.request) || caches.match('/login');
        })
    );
  } else {
    // For images/CSS, keep using the cache to save data
    event.respondWith(
      caches.match(event.request).then(response => {
        return response || fetch(event.request);
      })
    );
  }
});
