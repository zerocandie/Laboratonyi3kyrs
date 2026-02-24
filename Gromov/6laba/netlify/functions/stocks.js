// netlify/functions/stocks.js

// Начальные данные (аналог stocks.json)
let stocks = [
  {
    id: 1,
    src: "https://i.pinimg.com/originals/c9/ea/65/c9ea654eb3a7398b1f702c758c1c4206.jpg",
    title: "Акция",
    text: "Такой акции вы еще не видели 1"
  },
  {
    id: 2,
    src: "https://i.pinimg.com/originals/c9/ea/65/c9ea654eb3a7398b1f702c758c1c4206.jpg",
    title: "Акция",
    text: "Такой акции вы еще не видели 2"
  },
  {
    id: 3,
    src: "https://i.pinimg.com/originals/c9/ea/65/c9ea654eb3a7398b1f702c758c1c4206.jpg",
    title: "Акция",
    text: "Такой акции вы еще не видели 3"
  }
];

let nextId = 4; // следующий ID для новых акций

export default async (request) => {
  // Парсим URL и метод
  const url = new URL(request.url);
  const path = url.pathname; // например: /stocks или /stocks/2
  const method = request.method;

  // Извлекаем ID из пути: /stocks/2 → id = "2"
  const parts = path.split('/').filter(p => p);
  const id = parts[1] ? parseInt(parts[1], 10) : null;

  try {
    // === GET ===
    if (method === 'GET') {
      if (id !== null) {
        // GET /stocks/2
        const stock = stocks.find(s => s.id === id);
        if (!stock) {
          return new Response(JSON.stringify({ error: 'Акция не найдена' }), {
            status: 404,
            headers: { 'Content-Type': 'application/json' }
          });
        }
        return new Response(JSON.stringify(stock), {
          headers: { 'Content-Type': 'application/json' }
        });
      } else {
        // GET /stocks
        return new Response(JSON.stringify(stocks), {
          headers: { 'Content-Type': 'application/json' }
        });
      }
    }

    // === POST ===
    if (method === 'POST') {
      // Читаем тело запроса
      let body = '';
      for await (const chunk of request.body) {
        body += chunk.toString();
      }

      let stock;
      try {
        stock = JSON.parse(body);
      } catch (e) {
        return new Response(JSON.stringify({ error: 'Неверный JSON' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      // Валидация
      if (
        typeof stock.title !== 'string' ||
        typeof stock.text !== 'string' ||
        typeof stock.src !== 'string'
      ) {
        return new Response(JSON.stringify({ error: 'Неверный формат акции' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      // Присваиваем новый ID
      const newStock = {
        id: nextId++,
        title: stock.title,
        text: stock.text,
        src: stock.src
      };

      stocks.push(newStock);

      return new Response(JSON.stringify(newStock), {
        status: 201,
        headers: { 'Content:Type': 'application/json' }
      });
    }

    // === DELETE ===
    if (method === 'DELETE' && id !== null) {
      const index = stocks.findIndex(s => s.id === id);
      if (index === -1) {
        return new Response(JSON.stringify({ error: 'Акция не найдена' }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      stocks.splice(index, 1);

      return new Response(JSON.stringify({ message: 'Акция удалена' }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // === Метод не поддерживается ===
    return new Response(JSON.stringify({ error: 'Метод не разрешён' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json', 'Allow': 'GET, POST, DELETE' }
    });

  } catch (err) {
    console.error('Ошибка в stocks.js:', err);
    return new Response(JSON.stringify({ error: 'Внутренняя ошибка' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};