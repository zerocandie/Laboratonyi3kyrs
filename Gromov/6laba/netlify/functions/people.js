// netlify/functions/people.js
let people = [
  {
    id: "1",
    name: "Сныткин Илья",
    role: "Студент ИС-23",
    photo: "https://i.pravatar.cc/300?img=1",
    description: "Любит Docker и киберпанк."
  },
  {
    id: "2",
    name: "Иван",
    role: "Студент",
    photo: "https://i.pravatar.cc/300?img=2",
    description: "Крутой чувак"
  }
];

let nextId = 3;

export default async (request) => {
  const url = new URL(request.url);
  const path = url.pathname;
  const method = request.method;

  const parts = path.split('/').filter(p => p);
  const id = parts[1];

  // Только для демонстрации — данные не сохраняются после перезагрузки функции!
  // В реальном проекте нужно использовать внешнюю БД.

  if (method === 'GET') {
    if (id) {
      const person = people.find(p => p.id === id);
      if (!person) {
        return new Response(JSON.stringify({ error: 'Not found' }), { status: 404 });
      }
      return new Response(JSON.stringify(person));
    }
    return new Response(JSON.stringify(people));
  }

  if (method === 'POST') {
    let body = '';
    for await (const chunk of request.body) {
      body += chunk.toString();
    }
    const person = JSON.parse(body);
    person.id = String(nextId++);
    people.push(person);
    return new Response(JSON.stringify(person), { status: 201 });
  }

  if (method === 'DELETE' && id) {
    const index = people.findIndex(p => p.id === id);
    if (index === -1) {
      return new Response(JSON.stringify({ error: 'Not found' }), { status: 404 });
    }
    people.splice(index, 1);
    return new Response(JSON.stringify({ message: 'Deleted' }));
  }

  return new Response(JSON.stringify({ error: 'Method not allowed' }), { status: 405 });
};