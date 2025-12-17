// front/js/pages/add-person/index.js
class AddPersonPage {
    constructor(parent) {
        this.parent = parent;
    }

    render() {
        this.parent.innerHTML = `
            <div class="container mt-4">
                <h2>Добавить нового человека</h2>
                <form id="add-person-form">
                    <div class="mb-3">
                        <label class="form-label">Имя</label>
                        <input type="text" class="form-control" name="name" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Роль</label>
                        <input type="text" class="form-control" name="role" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Фото</label>
                        <input type="file" class="form-control" name="photo" accept="image/*" required>
                        <div class="mt-2" id="photo-preview"></div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Описание</label>
                        <textarea class="form-control" name="description" rows="3" required></textarea>
                    </div>
                    <button type="submit" class="btn btn-success">✅ Сохранить</button>
                    <button type="button" class="btn btn-secondary ms-2" id="back-btn">← Назад</button>
                </form>
            </div>
        `;

        // Превью фото
        const fileInput = document.querySelector('input[name="photo"]');
        const preview = document.getElementById('photo-preview');
        fileInput.addEventListener('change', () => {
            preview.innerHTML = '';
            if (fileInput.files[0]) {
                const url = URL.createObjectURL(fileInput.files[0]);
                preview.innerHTML = `<img src="${url}" class="img-thumbnail" style="max-height:200px;">`;
            }
        });

        document.getElementById('back-btn').addEventListener('click', () => {
            window.location.hash = '#people';
        });

        // Основной обработчик формы
        document.getElementById('add-person-form').addEventListener('submit', async (e) => {
            e.preventDefault();

            const name = document.querySelector('input[name="name"]').value.trim();
            const role = document.querySelector('input[name="role"]').value.trim();
            const description = document.querySelector('textarea[name="description"]').value.trim();
            const photoFile = fileInput.files[0];

            if (!name || !role || !description || !photoFile) {
                alert('Заполните все поля и выберите фото');
                return;
            }

            try {
                // Конвертируем фото в base64
                const photoBase64 = await this.fileToBase64(photoFile);

                // Отправляем JSON
                const response = await fetch('/api/people', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        name,
                        role,
                        description,
                        photoBase64 // ← ключевое изменение!
                    })
                });

                if (response.ok) {
                    alert('Человек успешно добавлен!');
                    window.location.hash = '#people';
                } else {
                    const error = await response.json();
                    alert('Ошибка: ' + (error.error || 'неизвестная ошибка'));
                }
            } catch (err) {
                alert('Не удалось добавить: ' + err.message);
            }
        });
    }

    // Вспомогательная функция: файл → base64
    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result); // data:image/png;base64,...
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }
}

export default AddPersonPage;