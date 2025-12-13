// API/server.js
const express = require('express');
const path = require('path');
const multer = require('multer');

// Импорты роутеров
const peopleRouter = require('./internal/people/router');


// === Multer: сохранение фото в front/assets/ ===
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, path.join(__dirname, '..', 'front', 'assets'));
    },
    filename: (req, file, cb) => {
        const ext = path.extname(file.originalname).toLowerCase();
        const name = Date.now() + ext;
        cb(null, name);
    }
});

const upload = multer({
    storage,
    fileFilter: (req, file, cb) => {
        if (!file.mimetype.startsWith('image/')) {
            return cb(new Error('Разрешены только изображения'), false);
        }
        cb(null, true);
    },
    limits: { fileSize: 5 * 1024 * 1024 } // 5 МБ
});

// === Создание app ===
const app = express();

app.use(express.json());
app.use(express.static(path.join(__dirname, '..', 'front')));

// === Кастомный POST-роут для людей с загрузкой фото ===
app.post('/api/people', upload.single('photo'), (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'Фото обязательно' });
        }

        const { PeopleService } = require('./internal/people/PeopleService');

        const person = {
            id: Date.now().toString(),
            name: req.body.name,
            role: req.body.role,
            description: req.body.description,
            photo: `assets/${req.file.filename}` // путь для фронтенда
        };

        const created = PeopleService.create(person);
        res.status(201).json(created);
    } catch (err) {
        console.error('Ошибка добавления:', err);
        res.status(400).json({ error: err.message });
    }
});

// Остальные роуты
app.use('/api/people', peopleRouter); // ← этот роут теперь используется ТОЛЬКО для GET, DELETE


// Запуск
const PORT = 8000;
app.listen(PORT, 'localhost', () => {
    console.log(`✅ Сервер запущен: http://localhost:${PORT}`);
});