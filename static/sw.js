const CACHE_NAME = 'coreefx-v2'; // Changed version to force an update
const urlsToCache = [
  '/',
  '/static/images/brain.png',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
  'https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&family=Roboto:wght@400;500;700&display=swap'
];

// 1. Install Event - Saves files to the phone
self.addEventListener('install', event => {
  self.skipWaiting(); // Forces the new service worker to become active immediately
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('Opened cache');
      return cache.addAll(urlsToCache);
    })
  );
});

// 2. Activate Event - Cleans up old versions and removes the "loading" feeling
self.addEventListener('activate', event => {
  event.waitUntil(
    Promise.all([
      clients.claim(), // Takes control of the page immediately
      caches.keys().then(cacheNames => {
        return Promise.all(
          cacheNames.map(cache => {
            if (cache !== CACHE_NAME) {
              return caches.delete(cache);
            }
          })
        );
      })
    ])
  );
});

// 3. Fetch Event - The "Offline First" strategy to remove the horizontal line
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      // Return cached file if found, otherwise fetch from network
      return response || fetch(event.request);
    })
  );
});
