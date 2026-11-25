FROM node:20-alpine

WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev

COPY src ./src
COPY tests ./tests

RUN node --test tests/*.test.js

EXPOSE 3000
CMD ["node", "src/index.js"]
