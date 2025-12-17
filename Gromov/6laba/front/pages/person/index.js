class PersonPage {
    constructor(parent) {
        this.parent = parent;
    }

    async render() {
    // Правильно извлекаем id из #person/123
    const hash = window.location.hash.substring(1); // убираем #
    const parts = hash.split('/');
    const id = parts[1]; // ["person", "1"] → parts[1] = "1"

    if (!id || id.trim() === '') {
        this.parent.innerHTML = `<p class="text-danger">Неверный или отсутствующий ID пользователя</p>`;
        return;
    }

    this.parent.innerHTML = '<p>Загрузка...</p>';

    try {
        const res = await fetch(`/api/people/${encodeURIComponent(id)}`);
        if (!res.ok) throw new Error('Пользователь не найден');
        const person = await res.json();

        this.parent.innerHTML = `
            <div class="container mt-4">
                <h1>${person.name}</h1>
                <img src="${person.photo || 'assets/default.jpg'}" class="img-fluid rounded mb-3" alt="${person.name}">
                <p><strong>Роль:</strong> ${person.role}</p>
                <p>${person.description}</p>
                <button class="btn btn-secondary" onclick="window.location.hash='#people'">← Назад</button>
            </div>
        `;
    } catch (err) {
        this.parent.innerHTML = `<p class="text-danger">Ошибка: ${err.message}</p>`;
    }
}
}

export default PersonPage;