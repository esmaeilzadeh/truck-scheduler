# Project Guidelines Skill - AI Chat Read API

This skill documents the conventions and patterns for this NestJS read-only API.

---

## When to Use

Reference this skill when working on this repository. It covers:
- Architecture overview
- File structure
- Code patterns
- Testing and local run workflow
- Environment variables

---

## Architecture Overview

**Tech Stack:**
- **Backend**: NestJS 10 + TypeScript
- **Database**: PostgreSQL via Drizzle ORM
- **Messaging**: RabbitMQ (nestjs-rabbitmq)
- **Queue**: Bull (failed log retries)
- **Cache**: Redis
- **Auth**: API key guard via gRPC
- **Observability**: Pino logging + OpenTelemetry
- **Docs**: Swagger (non-production)

**Services:**
```
┌─────────────────────────────────────────────────────────────┐
│                     AI Chat Read API                        │
│  NestJS + Drizzle + Swagger                                 │
└─────────────────────────────────────────────────────────────┘
             │          │            │           │
             ▼          ▼            ▼           ▼
        PostgreSQL    Redis       RabbitMQ     gRPC
                                    │      (API key)
                                    ▼
                                 Bull
```

---

## File Structure

```
src/
├── application/
│   ├── middlewares/       # request-id middleware
│   ├── services/proxies/  # gRPC + logging proxies
│   └── use-cases/          # orchestration (execute methods)
├── domain/
│   ├── models/             # entities + Drizzle schemas
│   └── services/           # repositories, mappers, factories
├── infrastructure/
│   ├── config/             # env-driven configuration
│   ├── constants/          # defaults + constant values
│   ├── database/           # DrizzleService
│   ├── decorators/         # custom metadata
│   ├── modules/            # Nest modules
│   └── utilities/          # helpers and error utils
└── presentation/
    ├── controllers/        # HTTP endpoints
    ├── DTOs/               # validation + swagger docs
    ├── filters/            # GlobalExceptionFilter
    ├── guards/             # ApiKeyGuard
    └── interceptors/       # request logging
```

---

## Code Patterns

### Controllers + Guards

- Controllers use DTOs for validation and `ApiKeyGuard` for access control.
- Each handler passes the DTO to a use case.

```typescript
@Get('search')
@ApiSecurity('apiKey')
@KeyGuardMetadata({ scope: 'chathub-read', resource: 'messages', action: 'read' })
@UseGuards(ApiKeyGuard)
search(@Query() q: MessageSearchDto) {
  return this.searchMessages.execute(q);
}
```

### Use Cases

- Use cases are thin orchestration layers with `execute()`.

```typescript
@Injectable()
export class SearchMessagesUseCase {
  constructor(@Inject('MessageRepository') private readonly repo: MessageRepository) {}
  async execute(query: MessageSearchDto) {
    return this.repo.searchByContent(query);
  }
}
```

### Repositories + Mappers

- Repository interfaces live in `domain/services/repositories`.
- Drizzle implementations live beside the interfaces and are registered by token.
- Mappers translate Drizzle records to entities.

```typescript
export const repositories: Provider[] = [
  { provide: 'ConversationRepository', useClass: DrizzleConversationRepository },
  { provide: 'MessageRepository', useClass: DrizzleMessageRepository },
];
```

### Cursor Pagination + Search

- Cursor pagination uses `id` as cursor with `limit` and `nextCursor`.
- Message search uses `to_tsvector` + `to_tsquery` with `term1 & term2` format.
- Conversation search checks related messages with full-text search in a subquery.

---

## Testing and Local Run

```bash
# Local dev
npm run start:dev

# Tests
npm test
npm run test:watch
npm run test:cov

# Integration tests (isolated DB)
npm run test:integration:env:up
npm run test:integration:ci
npm run test:integration:env:down
```

---

## Environment Variables

```bash
# App
PORT=3000
SERVICE_NAME=sahab-chat-hub-read
NODE_ENV=development

# Database
DATABASE_URL=postgresql://...
TEST_DATABASE_URL=postgresql://...

# Redis
REDIS_URL=redis://localhost:6379
REDIS_PREFIX=__sahab-chat-hub-read
REDIS_PING_INTERVAL_SECONDS=5

# RabbitMQ
RMQ_URI=amqp://guest:guest@localhost:5672
RMQ_PREFETCH_COUNT=20
RMQ_MESSAGE_QUEUE=chathub-read-queue
RMQ_ROUTING_KEY=chathub-read-routing-key
RMQ_EXCHANGE=chathub-read-exchange

# gRPC
API_KEY_MANAGEMENT_GRPC_URL=localhost:3100

# Logging
PART_LOGGING_URL=http://...
ACTIVE_REQUEST_LOGGER=true
```

---

## Project Conventions

- Keep controllers thin; all data access goes through repositories.
- Always validate input with DTOs and `class-validator`.
- Apply `ApiKeyGuard` and `KeyGuardMetadata` to protected endpoints.
- Use `HttpLoggerInterceptor` for request logging; rely on global exception filter.
- Use Drizzle schemas in `domain/models/schemas` for DB structure.

---

## Related Skills

- `coding-standards.md` - General coding best practices
- `backend-patterns.md` - API and database patterns
