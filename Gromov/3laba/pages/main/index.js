// MainPage.js
import { AccordionComponent } from "../../components/accordion/index.js";

export class MainPage {
    constructor(parent, onDogSelect, onAddDog) {
        this.parent = parent;
        this.onDogSelect = onDogSelect;
        this.onAddDog = onAddDog;
    }

    async fetchDogs() {
        const res = await fetch('/api/dogs');
        if (!res.ok) throw new Error('Не удалось загрузить породы');
        return await res.json();
    }

    async render() {
        this.parent.innerHTML = '';

        // Кнопка добавления
        const addButton = document.createElement('button');
        addButton.className = 'btn btn-success mb-3';
        addButton.textContent = '➕ Добавить породу';
        addButton.addEventListener('click', () => this.onAddDog());
        this.parent.append(addButton);

        this.parent.innerHTML += '<h1 class="text-center my-3">Породы собак</h1>';

        try {
            const dogs = await this.fetchDogs();

            if (dogs.length === 0) {
                this.parent.innerHTML += '<p class="text-center text-muted">Нет пород</p>';
                return;
            }

            const accordionContainer = document.createElement('div');
            this.parent.append(accordionContainer);

            const items = dogs.map(dog => ({
                id: dog.id,
                title: dog.title,
                content: `
                    <img src="${dog.img}" class="dog-card-img" alt="${dog.title}">
                    <div class="dog-content">
                        <p class="breed-desc">${dog.description}</p>
                        <div>
                            <button type="button" class="btn btn-info details-btn me-2" data-id="${dog.id}">Подробнее</button>
                            <button type="button" class="btn btn-danger delete-btn" data-id="${dog.id}">🗑 Удалить</button>
                        </div>
                    </div>
                `
            }));

            const accordion = new AccordionComponent(accordionContainer, items, (dogId) => {
                this.onDogSelect(dogId);
            });

            accordion.render();

            // Навесить обработчики удаления
            accordionContainer.querySelectorAll('.delete-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const id = btn.dataset.id;
                    if (confirm(`Удалить породу "${id}"?`)) {
                        try {
                            const res = await fetch(`/api/dogs/${id}`, { method: 'DELETE' });
                            if (res.ok) {
                                this.render(); // перерисовать список
                            } else {
                                alert('Ошибка при удалении');
                            }
                        } catch (err) {
                            alert('Не удалось удалить: ' + err.message);
                        }
                    }
                });
            });

        } catch (err) {
            this.parent.innerHTML = `<div class="alert alert-danger">Ошибка: ${err.message}</div>`;
        }
    }
}