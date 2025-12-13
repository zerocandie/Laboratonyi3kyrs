import MainPage from './pages/people-main/index.js';
import PersonPage from './pages/person/index.js';
import AddPersonPage from './pages/add-person/index.js';

class App {
    constructor() {
        this.root = document.getElementById('app');
        this.currentPage = null;
        this.init();
    }

    init() {
        window.addEventListener('hashchange', this.handleRoute.bind(this));
        this.handleRoute(); // Загрузка при старте
    }

    // В классе App
handleRoute() {
    const hash = window.location.hash;

    if (hash === '' || hash === '#people' || hash === '#') {
        this.showMainPage();
    } else if (hash.startsWith('#person/')) {
        this.showPersonPage();
    } else if (hash === '#add-person') {
    this.showAddPersonPage();
    } else {
        this.showMainPage();
    }
}

showAddPersonPage() {
    if (this.currentPage) this.currentPage = null;
    this.currentPage = new AddPersonPage(this.root);
    this.currentPage.render();
}

    showMainPage() {
        if (this.currentPage) this.currentPage = null;
        this.currentPage = new MainPage(this.root);
        this.currentPage.render();
    }

    showPersonPage() {
        if (this.currentPage) this.currentPage = null;
        this.currentPage = new PersonPage(this.root);
        this.currentPage.render();
    }
}

new App();