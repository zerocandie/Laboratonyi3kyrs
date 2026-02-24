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
                        <label class="form-label">Ссылка на фото</label>
                        <input type="url" class="form-control" name="photo" placeholder="https://example.com/photo.jpg" required>
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

        document.getElementById('back-btn').addEventListener('click', () => {
            window.location.hash = '#people';
        });

        document.getElementById('add-person-form').addEventListener('submit', async (e) => {
            e.preventDefault();

            const name = document.querySelector('input[name="name"]').value.trim();
            const role = document.querySelector('input[name="role"]').value.trim();
            const photo = document.querySelector('input[name="photo"]').value.trim();
            const description = document.querySelector('textarea[name="description"]').value.trim();

            if (!name || !role || !photo || !description) {
                alert('Заполните все поля');
                return;
            }

            try {
                const response = await fetch('/api/people', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, role, photo, description })
                });

                if (response.ok) {
                    alert('✅ Человек добавлен!');
                    window.location.hash = '#people';
                } else {
                    const error = await response.json();
                    alert('❌ Ошибка: ' + (error.error || 'неизвестно'));
                }
            } catch (err) {
                alert('⚠️ Не удалось добавить: ' + err.message);
            }
        });
    }
}

export default AddPersonPage;