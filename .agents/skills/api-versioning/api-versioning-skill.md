## When to Activate:
Apply these versioning rules in **any** of these situations:
- Starting a new API project, designing the first public endpoints.
- Introducing a change to an existing API: adding/removing fields, changing field types or semantics, altering error structures, authentication, or response shapes.
- Performing code review on pull requests that touch API route definitions, serializers/DTOs, middleware, or error handling.
- Planning the deprecation and removal of an older API version.
- Writing or updating API documentation (OpenAPI/Swagger) or client SDKs.
- Setting up automated testing, contract testing, or monitoring across API versions.
- Onboarding team members to API development standards.
- Any discussion about whether a change is “breaking” or “non-breaking”.

If a task does not affect external API contracts (e.g., purely internal refactoring behind the same version boundary), the versioning rules for breaking changes do not apply, but code organization, documentation, and testing principles must still be followed.

### 1. Versioning Strategy
- **Default approach**: Use URL path versioning (e.g., `/api/v1/resource`). This is the most transparent, easily cacheable, and compatible with all HTTP clients.
- **Alternative (if justified)**: For minor, non-breaking additions (new optional fields, new endpoints), you may use a request header (`Accept-Version: v1` or a custom `X-API-Version` header) **only** when backwards compatibility is fully maintained. Never use query parameters for versioning (`?version=1`); they pollute the resource identity and complicate caching.
- **Internal services**: Prefer semantic versioning conveyed via content negotiation or dedicated headers, but keep the external API consistent with path versioning.

### 2. Version Naming and Granularity
- Use simple, incremental major versions: `v1`, `v2`, `v3`… for the public API. Never expose minor/patch details (e.g., `v1.2`) in the URI.
- A new version is created only when you introduce **breaking changes** (removing/renaming fields, changing field types, altering error structures, modifying authentication requirements).
- Non-breaking additions (new endpoints, new optional fields) must be added to the **current** version without incrementing the version number.

### 3. Backwards Compatibility
- Existing endpoints and response structures must remain unchanged for the entire lifecycle of a version. Once released, a version is immutable.
- When adding a field, always make it optional and provide a safe default. Clients written against the older version must continue to work without modification.
- Never silently change the meaning of an existing field. If the semantics shift, that is a breaking change and requires a new version.

### 4. Code Organization
- **Route isolation**: Version-specific logic must reside in separate modules, controllers, or packages (e.g., `controllers/v1/`, `controllers/v2/`). Shared business logic should be extracted into a common layer to avoid duplication.
- **Middleware/Dispatch**: Use a routing middleware that selects the correct handler based on the version prefix. The main router should remain clean:
  ```python
  # Example in FastAPI / Flask / Express style
  app.mount("/api/v1", v1_router)
  app.mount("/api/v2", v2_router)
  ```
- **Request/Response models**: Each version owns its own DTOs/serializers to prevent unintended changes from leaking across versions.

### 5. Deprecation and Sunset Policy
- When a new version is introduced, the previous version(s) enter a **deprecation period**. You must:
    - Add a `Deprecation` HTTP header to all responses from the old version: `Deprecation: true`
    - Add a `Sunset` header with the date (in HTTP-date format) when the version will be turned off: `Sunset: Sat, 31 Dec 2025 23:59:59 GMT`
    - Log a warning message on the server whenever a deprecated endpoint is accessed.
- Never remove a deprecated version without a publicly communicated timeline and active client migration support. The deprecation window should be at least 6–12 months for external APIs.

### 6. Documentation
- Every versioned endpoint must have corresponding OpenAPI/Swagger documentation, clearly separated per version (e.g., `/docs/v1`, `/docs/v2`).
- API documentation must explicitly state which version is stable, which are deprecated, and the sunset date.
- Include a changelog that maps each version change to the exact differences (added/removed fields, new endpoints, behavioral changes).

### 7. Testing
- Maintain a comprehensive test suite for **each active version**. Never delete tests for a version until it is fully sunset.
- Use contract tests to verify that the response shape of an older version does not accidentally change when shared code is modified.
- Integration tests must run against all currently supported versions.

### 8. Client and SDK Considerations
- When generating client SDKs, version them to match the API (e.g., `client-v1`, `client-v2`). No SDK should default to the "latest" unstable version in production.
- Server code must never silently upgrade a client request to a newer version; it must respect the client’s chosen version.

### 9. Error Handling
- Version-specific error responses must remain consistent. Adding a new error code is a non-breaking change, but changing the structure of an error body is breaking and requires a new version.
- If an endpoint is removed in a newer version, the server must return `410 Gone` (not `404 Not Found`) and include a message pointing to the migration guide.

### 10. Practical Implementation Checklist
When you write or review code touching API endpoints, verify the following:
- [ ] New breaking change → new major version path created, old version unchanged.
- [ ] Non-breaking addition → added to the current version only.
- [ ] Routes, controllers, models are physically separated by version.
- [ ] Deprecated versions return `Deprecation` + `Sunset` headers.
- [ ] Documentation updated per version.
- [ ] Tests exist for each active version.
- [ ] No version wildcards (`/api/v{version}/`) or version detection from subtle client clues – always explicit.

Adhere to these rules rigorously. If a trade-off must be made, explain the reasoning in a comment, and always lean toward explicit, stable, and well-documented version boundaries.
```