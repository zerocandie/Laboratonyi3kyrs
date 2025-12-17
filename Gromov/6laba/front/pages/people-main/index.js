// front/pages/people-main/index.js
class MainPage {
    constructor(parent) {
        this.parent = parent;
    }

    async render() {
        try {
            const res = await fetch('/api/people');
            if (!res.ok) throw new Error('Не удалось загрузить список');
            const people = await res.json();

            this.parent.innerHTML = `
                <h1 class="my-4">Список участников</h1>
                <div id="people-list" class="row row-cols-1 row-cols-md-3 g-4"></div>
                <button class="btn btn-success mt-3" onclick="window.location.hash='#add-person'">
                    ➕ Добавить
                </button>
            `;

            const list = document.getElementById('people-list');

            // Фильтруем только тех, у кого есть id
            const validPeople = people.filter(p => p.id != null);

            if (validPeople.length === 0) {
                list.innerHTML = '<p class="text-muted">Нет участников с корректным ID</p>';
                return;
            }

            validPeople.forEach(p => {
                const col = document.createElement('div');
                col.className = 'col';
                col.innerHTML = `
                    <div class="card h-100">
                        <img 
                            src="${p.photo || 'assets/default.jpg'}" 
                            class="card-img-top" 
                            alt="${p.name || 'Без имени'}"
                            onerror="this.src='assets/default.jpg'"
                        >
                        <div class="card-body">
                            <h5 class="card-title">${p.name || '—'}</h5>
                            <p class="card-text">${p.role || '—'}</p>
                            <button 
                                class="btn btn-primary btn-sm me-2" 
                                onclick="window.location.hash='#person/${encodeURIComponent(p.id)}'"
                            >
                                Подробнее
                            </button>
                            <button 
                                class="btn btn-danger btn-sm delete-btn" 
                                data-id="${p.id}"
                            >
                                Удалить
                            </button>
                        </div>
                    </div>
                `;
                list.appendChild(col);
            });

            // Обработчик удаления
            list.querySelectorAll('.delete-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const id = btn.dataset.id;
                    if (!id || !confirm(`Удалить участника ${id}?`)) return;

                    try {
                        const res = await fetch(`/api/people/${encodeURIComponent(id)}`, {
                            method: 'DELETE'
                        });
                        if (res.ok) {
                            this.render(); // перезагрузить список
                        } else {
                            alert('Ошибка при удалении');
                        }
                    } catch (err) {
                        console.error(err);
                        alert('Не удалось удалить: ' + err.message);
                    }
                });
            });

        } catch (err) {
            this.parent.innerHTML = `
                <div class="alert alert-danger">
                    Ошибка загрузки: ${err.message}
                </div>
            `;
        }
    }
}

export default MainPage;