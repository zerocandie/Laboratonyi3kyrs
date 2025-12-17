const { DogsRepository } = require('./DogsRepository.js');

class DogDAO {
    constructor(id, title, img, description) {
        this.id = id;
        this.title = title;
        this.img = img;
        this.description = description;
    }

    static _validateId(id) {
        if (!id || typeof id !== 'string') {
            throw new Error('ID породы должен быть непустой строкой');
        }
        return id;
    }

    static _validate(dog) {
        if (!dog.id || !dog.title || !dog.img || !dog.description) {
            throw new Error('Все поля обязательны: id, title, img, description');
        }
        this._validateId(dog.id);
    }

    static findAll() {
        const dogs = DogsRepository.read();
        return dogs.map(d => new this(d.id, d.title, d.img, d.description));
    }

    static findById(id) {
        this._validateId(id);
        const dogs = DogsRepository.read();
        const dog = dogs.find(d => d.id === id);
        if (!dog) throw new Error(`Порода с id "${id}" не найдена`);
        return new this(dog.id, dog.title, dog.img, dog.description);
    }

    static create(dog) {
        this._validate(dog);
        const dogs = DogsRepository.read();
        if (dogs.some(d => d.id === dog.id)) {
            throw new Error(`Порода с id "${dog.id}" уже существует`);
        }
        DogsRepository.write([...dogs, dog]);
        return new this(dog.id, dog.title, dog.img, dog.description);
    }

    static delete(id) {
        this._validateId(id);
        const dogs = DogsRepository.read();
        const filtered = dogs.filter(d => d.id !== id);
        if (filtered.length === dogs.length) {
            throw new Error(`Порода с id "${id}" не найдена`);
        }
        DogsRepository.write(filtered);
        return true;
    }

    toJSON() {
        return {
            id: this.id,
            title: this.title,
            img: this.img,
            description: this.description
        };
    }
}

module.exports = { DogDAO };