// main.js (точка входа фронтенда)
import { MainPage } from './pages/MainPage.js';
import { DogDetailsPage } from './pages/DogDetailsPage.js';
import { AddDogPage } from './pages/AddDogPage.js';

const app = document.getElementById('app');

let currentPage = null;

const showMain = () => {
    if (currentPage) currentPage = null;
    currentPage = new MainPage(app, showDetails, showAddDog);
    currentPage.render();
};

const showDetails = (dogId) => {
    currentPage = new DogDetailsPage(app, showMain);
    currentPage.render(dogId);
};

const showAddDog = () => {
    currentPage = new AddDogPage(app, showMain);
    currentPage.render();
};

// Запуск
showMain();