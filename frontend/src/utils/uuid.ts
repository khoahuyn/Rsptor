import { v4 as uuidv4, v1 as uuidv1, validate as uuidValidate, version as uuidVersion } from 'uuid'

/**
 * UUID utility functions for generating and validating UUIDs
 */
export class UuidUtils {
  /**
   * Generate a random UUID v4
   * @returns A new UUID v4 string
   */
  static generateV4(): string {
    return uuidv4()
  }

  /**
   * Generate a timestamp-based UUID v1
   * @returns A new UUID v1 string
   */
  static generateV1(): string {
    return uuidv1()
  }

  /**
   * Validate if a string is a valid UUID
   * @param uuid The string to validate
   * @returns True if valid UUID, false otherwise
   */
  static isValid(uuid: string): boolean {
    return uuidValidate(uuid)
  }

  /**
   * Get the version of a UUID
   * @param uuid The UUID string
   * @returns The version number or null if invalid
   */
  static getVersion(uuid: string): number | null {
    if (!this.isValid(uuid)) {
      return null
    }
    return uuidVersion(uuid)
  }

  /**
   * Generate a short UUID for display purposes (first 8 characters)
   * @returns Short UUID string
   */
  static generateShort(): string {
    return uuidv4().substring(0, 8)
  }

  /**
   * Generate a UUID for knowledge base IDs
   * Uses UUID v4 for random generation
   * @returns Knowledge base ID
   */
  static generateKnowledgeBaseId(): string {
    return uuidv4()
  }

  /**
   * Generate a UUID for document IDs
   * Uses UUID v4 for random generation
   * @returns Document ID
   */
  static generateDocumentId(): string {
    return uuidv4()
  }

  /**
   * Generate a UUID for session IDs
   * Uses UUID v4 for random generation
   * @returns Session ID
   */
  static generateSessionId(): string {
    return uuidv4()
  }

  /**
   * Generate a UUID for message IDs
   * Uses UUID v4 for random generation
   * @returns Message ID
   */
  static generateMessageId(): string {
    return uuidv4()
  }

  /**
   * Format UUID for display (add hyphens if missing)
   * @param uuid UUID string with or without hyphens
   * @returns Properly formatted UUID string
   */
  static formatForDisplay(uuid: string): string {
    if (uuid.includes('-')) {
      return uuid
    }
    
    // Add hyphens to UUID without them (32 character string)
    if (uuid.length === 32) {
      return `${uuid.substring(0, 8)}-${uuid.substring(8, 12)}-${uuid.substring(12, 16)}-${uuid.substring(16, 20)}-${uuid.substring(20)}`
    }
    
    return uuid
  }

  /**
   * Remove hyphens from UUID for compact storage
   * @param uuid UUID string with hyphens
   * @returns UUID string without hyphens
   */
  static removeHyphens(uuid: string): string {
    return uuid.replace(/-/g, '')
  }
}