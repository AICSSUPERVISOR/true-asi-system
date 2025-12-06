/**
 * Company Save Test
 * 
 * Tests the saveCompany mutation to ensure it works correctly
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { appRouter } from './routers';
import type { Context } from './_core/context';

describe('Company Save Functionality', () => {
  const mockContext: Context = {
    user: {
      id: '1',
      openId: 'test-open-id',
      name: 'Test User',
      email: 'test@example.com',
      role: 'user',
    },
    req: {} as any,
    res: {} as any,
  };

  const mockBrregData = {
    organisasjonsnummer: '943574537',
    navn: 'CAPGEMINI NORGE AS',
    organisasjonsform: {
      kode: 'AS',
      beskrivelse: 'Aksjeselskap',
    },
    registreringsdatoEnhetsregisteret: '1995-02-19',
    naeringskode1: {
      kode: '62.200',
      beskrivelse: 'Konsulentvirksomhet tilknyttet informasjonsteknologi og forvaltning og drift av it-systemer',
    },
    antallAnsatte: 1557,
    forretningsadresse: {
      adresse: ['Karenslyst allé 20'],
      postnummer: '0278',
      poststed: 'OSLO',
      kommune: 'OSLO',
      kommunenummer: '0301',
    },
    registrertIMvaregisteret: true,
    registrertIForetaksregisteret: true,
    konkurs: false,
    underAvvikling: false,
  };

  it('should save company data successfully', async () => {
    const caller = appRouter.createCaller(mockContext);

    const result = await caller.brreg.saveCompany({
      orgnr: '943574537',
      brregData: mockBrregData,
    });

    expect(result.success).toBe(true);
    expect(result.companyId).toBeDefined();
    expect(result.companyId).toMatch(/^company_943574537_/);
  });

  it('should handle missing optional fields', async () => {
    const caller = appRouter.createCaller(mockContext);

    const minimalBrregData = {
      organisasjonsnummer: '123456789',
      navn: 'Test Company',
    };

    const result = await caller.brreg.saveCompany({
      orgnr: '123456789',
      brregData: minimalBrregData,
    });

    expect(result.success).toBe(true);
    expect(result.companyId).toBeDefined();
  });

  it('should fail without authentication', async () => {
    const unauthContext: Context = {
      user: null as any,
      req: {} as any,
      res: {} as any,
    };

    const caller = appRouter.createCaller(unauthContext);

    await expect(
      caller.brreg.saveCompany({
        orgnr: '943574537',
        brregData: mockBrregData,
      })
    ).rejects.toThrow();
  });
});
