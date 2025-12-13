// AddDogPage.js
export class AddDogPage {
    constructor(parent, onBack) {
        this.parent = parent;
        this.onBack = onBack;
    }

    render() {
        this.parent.innerHTML = `
            <div class="container mt-4">
                <h2>Добавить новую породу</h2>
                <form id="add-dog-form">
                    <div class="mb-3">
                        <label class="form-label">ID (латиницей, без пробелов)</label>
                        <input type="text" class="form-control" name="id" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Название</label>
                        <input type="text" class="form-control" name="title" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Путь к картинке (относительно /assets/)</label>
                        <input type="text" class="form-control" name="img" placeholder="Например: assets/Пудель.jpg" required>
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

        document.getElementById('back-btn').addEventListener('click', () => this.onBack());

        document.getElementById('add-dog-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const dog = {
                id: formData.get('id').trim(),
                title: formData.get('title').trim(),
                img: formData.get('img').trim(),
                description: formData.get('description').trim()
            };

            try {
                const res = await fetch('/api/dogs', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(dog)
                });

                if (res.ok) {
                    alert('Порода добавлена!');
                    this.onBack(); // вернуться к списку
                } else {
                    const err = await res.json();
                    alert('Ошибка: ' + err.error);
                }
            } catch (err) {
                alert('Не удалось добавить породу: ' + err.message);
            }
        });
    }
}