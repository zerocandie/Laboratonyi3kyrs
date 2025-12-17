// server.js
const express = require('express');
const dogsRouter = require('./internal/dogs/router');

const app = express();
const PORT = 8000;

app.use(express.json()); // важно для POST!
app.use(express.static('public'));

app.use('/api/dogs', dogsRouter); // ← все эндпоинты под /api/dogs

app.get('/', (req, res) => {
    res.sendFile(path.join(process.cwd(), 'index.html'));
});

app.listen(PORT, 'localhost', () => {
    console.log(`✅ Сервер запущен: http://localhost:${PORT}`);
});