export class AccordionComponent {
    constructor(parent, items, onItemClick) {
        this.parent = parent;
        this.items = items; 
        this.onItemClick = onItemClick; 
    }

    render() {
    const accordionHTML = this.items.map(item => `
        <div class="accordion-item">
            <h2 class="accordion-header" id="heading-${item.id}">
                <button class="accordion-button collapsed" type="button" data-id="${item.id}" 
                        data-bs-toggle="collapse" data-bs-target="#collapse-${item.id}" 
                        aria-expanded="false">
                    ${item.title}
                </button>
            </h2>
            <div id="collapse-${item.id}" class="accordion-collapse collapse" aria-labelledby="heading-${item.id}">
                <div class="accordion-body">
                    ${item.content}
                </div>
            </div>
        </div>
    `).join('');

    this.parent.innerHTML = `<div class="accordion">${accordionHTML}</div>`;

    
    this.parent.querySelectorAll('.details-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const id = e.target.dataset.id;
            this.onItemClick(id);
        });
    });
}
}