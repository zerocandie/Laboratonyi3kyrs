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

        document.getElementById('add-person-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);

            const name = formData.get('name').trim();
            const role = formData.get('role').trim();
            const description = formData.get('description').trim();
            const photo = formData.get('photo');

            if (!name || !role || !description || !photo) {
                alert('Заполните все поля');
                return;
            }

            const uploadData = new FormData();
            uploadData.append('name', name);
            uploadData.append('role', role);
            uploadData.append('description', description);
            uploadData.append('photo', photo);

            try {
                const res = await fetch('/api/people', {
                    method: 'POST',
                    body: uploadData 
                });

                if (res.ok) {
                    alert('Человек добавлен!');
                    window.location.hash = '#people';
                } else {
                    const err = await res.json();
                    alert('Ошибка: ' + (err.error || 'неизвестно'));
                }
            } catch (err) {
                alert('Не удалось добавить: ' + err.message);
            }
        });
    }
}

export default AddPersonPage;